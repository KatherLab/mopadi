"""

USAGE:
python convert_to_shards.py \
  --zip_glob "/data/*/*.zip" \
  --out_pattern "/out/{split}-{cohort}-%06d.tar" \
  --val_pct 0.10 \
  --workers 8
  
"""
import os, re, argparse, zipfile, glob
from typing import Optional, List
from tqdm import tqdm
import webdataset as wds
from multiprocessing import Process, Queue, cpu_count
from threading import Thread
import numpy as np
import json
import hashlib
from collections import defaultdict

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
STREAM_MAP = {"png": "png", "jpg": "jpg", "jpeg": "jpeg", "tif": "tif", "tiff": "tiff"}

def pick_val_patients_per_cohort(cohort2pids, val_pct: float):
    """
    cohort2pids: dict[str, list[str]]
    returns set of (cohort, pid) chosen for val
    """
    val_pairs = set()
    for cohort, pids in cohort2pids.items():
        uniq = sorted(set(pids))
        n = len(uniq)
        if n == 0:
            continue
        # deterministic: hash each patient within this cohort, sort by hash
        scored = []
        for pid in uniq:
            h = int(hashlib.md5(f"{cohort}|{pid}".encode("utf-8")).hexdigest(), 16)
            scored.append((h, pid))
        scored.sort(key=lambda x: x[0])

        k = max(1, int(round(val_pct * n))) if val_pct > 0 else 0
        k = min(k, n - 1) if n > 1 else (0 if val_pct < 0.5 else 1)
        for _, pid in scored[:k]:
            val_pairs.add((cohort, pid))
    return val_pairs

def patient_from_zip(basename: str, patient_re: Optional[re.Pattern] = None) -> str:
    pid = os.path.splitext(basename)[0]
    if patient_re:
        m = patient_re.search(pid)
        if m:
            pid = m.group(1) if m.groups() else m.group(0)
    return pid

def sanitize_stem_for_key(stem: str) -> str:
    stem = stem.replace(".", "_")
    stem = re.sub(r"[^\w\-\(\)]+", "_", stem)
    return stem.strip("_")

def fast_coords_bytes(coords_re: Optional[re.Pattern], stem: str):
    if not coords_re:
        return None
    m = coords_re.search(stem)
    if not m:
        return None
    # Cast to float32 first, then print with 9 significant digits (float32-safe)
    x32 = np.float32(float(m.group(1)))
    y32 = np.float32(float(m.group(2)))
    s = '{{"x":{:.9g},"y":{:.9g}}}'.format(float(x32), float(y32))
    return s.encode("utf-8")

def iter_zip_images(zpath: str, cohort: str, patient_id: str, coords_re: Optional[re.Pattern]):
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            name = info.filename
            lower = name.lower()
            if not lower.endswith(IMAGE_EXTS):
                continue
            data = zf.read(info)
            base = name.rsplit("/", 1)[-1]
            stem = base.rsplit(".", 1)[0]
            safe_stem = sanitize_stem_for_key(stem)
            coords_b = fast_coords_bytes(coords_re, stem)
            ext = lower.rsplit(".", 1)[-1]
            stream = STREAM_MAP.get(ext, "png")
            sample = {
                "__key__": f"{cohort}/{patient_id}/{safe_stem}",
                stream: data,
                "stem.txt": stem.encode("utf-8"),
                "patient.txt": patient_id.encode("utf-8"),
                "cohort.txt": cohort.encode("utf-8"),
            }
            if coords_b is not None:
                sample["coords.json"] = coords_b
            yield sample

def writer_loop(out_pattern: str, maxcount: int, q: Queue, total: int):
    assert "{split}" in out_pattern and "{cohort}" in out_pattern, \
        "--out_pattern must contain both '{split}' and '{cohort}'"
    writers = {}
    pbar = tqdm(total=total, desc="Writing shards")
    try:
        while True:
            item = q.get()
            if item is None:
                break
            split = item.pop("_split", "train")
            cohort = item.pop("_cohort", "unknown")
            key = (split, cohort)
            if key not in writers:
                pattern = out_pattern.format(split=split, cohort=cohort)
                os.makedirs(os.path.dirname(pattern), exist_ok=True)
                writers[key] = wds.ShardWriter(pattern, maxcount=maxcount)
            writers[key].write(item)
            pbar.update(1)
    finally:
        for w in writers.values():
            w.close()
        pbar.close()

def worker_reader(zips: List[str], cohorts: List[str], pids: List[str],
                  coords_regex: Optional[str], q: Queue, val_pairs: set):
    coords_re = re.compile(coords_regex) if coords_regex else None
    for zpath, cohort, pid in zip(zips, cohorts, pids):
        split = "val" if (cohort, pid) in val_pairs else "train"
        for sample in iter_zip_images(zpath, cohort, pid, coords_re):
            sample["_cohort"] = cohort
            sample["_split"] = split
            q.put(sample)

def chunk_indices(n_items: int, n_chunks: int):
    n_chunks = max(1, min(n_chunks, n_items))
    # near-even partitions
    base = n_items // n_chunks
    extra = n_items % n_chunks
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        yield range(start, start + size)
        start += size

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip_glob", required=True)
    p.add_argument("--out_pattern", required=True)
    p.add_argument("--cohort", default=None)
    p.add_argument("--maxcount", type=int, default=8000)
    p.add_argument("--coords_regex", default=r"tile_\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")
    p.add_argument("--patient_regex", default=None)
    p.add_argument("--workers", type=int, default=0, help=">0 to parallelize reading")
    p.add_argument("--queue_size", type=int, default=4096)
    p.add_argument("--val_pct", type=float, default=0.10,
               help="Validation fraction per cohort at patient level")
    args = p.parse_args()

    zips = sorted(glob.glob(args.zip_glob))
    if not zips:
        raise SystemExit(f"No zips matched {args.zip_glob}")

    # Precompile regexes once (used in single-process path and for computing patient ids)
    coords_re = re.compile(args.coords_regex) if args.coords_regex else None
    patient_re = re.compile(args.patient_regex) if args.patient_regex else None

    total = 0
    for z in tqdm(zips, desc="Prescanning zips"):
        with zipfile.ZipFile(z) as zf:
            # Using infolist() avoids building a separate namelist list of strings
            total += sum(info.filename.lower().endswith(IMAGE_EXTS) for info in zf.infolist())

    cohorts = [args.cohort or os.path.basename(os.path.dirname(z)) for z in zips]
    patient_ids = [patient_from_zip(os.path.basename(z), patient_re) for z in zips]

    # Build cohort -> patients map and choose val patients
    c2p = defaultdict(list)
    for coh, pid in zip(cohorts, patient_ids):
        c2p[coh].append(pid)
    val_pairs = pick_val_patients_per_cohort(c2p, args.val_pct)

    if val_pairs:
        c_train = defaultdict(set); c_val = defaultdict(set)
        for coh, pid in zip(cohorts, patient_ids):
            (c_val if (coh, pid) in val_pairs else c_train)[coh].add(pid)
        print("[split] cohorts:", len(set(cohorts)))
        for coh in sorted(set(cohorts)):
            print(f"[split] {coh}: train={len(c_train[coh])} patients  val={len(c_val[coh])} patients")

        out_dir = os.path.dirname(args.out_pattern.format(split="val", cohort="manifest"))
        os.makedirs(out_dir or ".", exist_ok=True)

        manifest_path = os.path.join(out_dir or ".", "split_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({"val": sorted(list(map(list, val_pairs)))}, f, indent=2)
        print(f"[split] wrote {manifest_path}")


    if args.workers <= 0:
        assert "{split}" in args.out_pattern and "{cohort}" in args.out_pattern, \
            "--out_pattern must contain '{split}' and '{cohort}'"
        writers = {}
        pbar = tqdm(total=total, desc="Writing shards")
        try:
            for zpath, cohort, pid in zip(zips, cohorts, patient_ids):
                split = "val" if (cohort, pid) in val_pairs else "train"
                key = (split, cohort)
                if key not in writers:
                    pattern = args.out_pattern.format(split=split, cohort=cohort)
                    os.makedirs(os.path.dirname(pattern), exist_ok=True)
                    writers[key] = wds.ShardWriter(pattern, maxcount=args.maxcount)
                for sample in iter_zip_images(zpath, cohort, pid, coords_re):
                    writers[key].write(sample)
                    pbar.update(1)
        finally:
            for w in writers.values():
                w.close()
            pbar.close()
        return

    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    q: Queue = Queue(maxsize=args.queue_size)

    writer = Thread(target=writer_loop,
                    args=(args.out_pattern, args.maxcount, q, total),
                    daemon=True)
    writer.start()

    procs: List[Process] = []
    for idxs in chunk_indices(len(zips), n_workers):
        pz = [zips[i] for i in idxs]
        pc = [cohorts[i] for i in idxs]
        pp = [patient_ids[i] for i in idxs]
        proc = Process(target=worker_reader,
                    args=(pz, pc, pp, args.coords_regex, q, val_pairs),
                    daemon=True)
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    q.put(None)
    writer.join()

if __name__ == "__main__":
    main()
