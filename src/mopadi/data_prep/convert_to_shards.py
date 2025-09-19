# convert_to_tar_shards_fast_prescan.py
import os, re, argparse, zipfile, glob
from typing import Optional, List
from tqdm import tqdm
import webdataset as wds
from multiprocessing import Process, Queue, cpu_count
from threading import Thread
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
STREAM_MAP = {"png": "png", "jpg": "jpg", "jpeg": "jpeg", "tif": "tif", "tiff": "tiff"}

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
            data = zf.read(info)  # zlib in C => GIL-friendly
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
            }
            if coords_b is not None:
                sample["coords.json"] = coords_b
            yield sample

def writer_loop(out_pattern: str, maxcount: int, q: Queue, total: int):
    with wds.ShardWriter(out_pattern, maxcount=maxcount) as sink:
        pbar = tqdm(total=total, desc="Writing shards")
        while True:
            item = q.get()
            if item is None:
                break
            sink.write(item)
            pbar.update(1)
        pbar.close()

def worker_reader(zips: List[str], cohorts: List[str], pids: List[str], coords_regex: Optional[str], q: Queue):
    coords_re = re.compile(coords_regex) if coords_regex else None
    for zpath, cohort, pid in zip(zips, cohorts, pids):
        for sample in iter_zip_images(zpath, cohort, pid, coords_re):
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

    if args.workers <= 0:
        with wds.ShardWriter(args.out_pattern, maxcount=args.maxcount) as sink:
            pbar = tqdm(total=total, desc="Writing shards")
            for zpath, cohort, pid in zip(zips, cohorts, patient_ids):
                for sample in iter_zip_images(zpath, cohort, pid, coords_re):
                    sink.write(sample)
                    pbar.update(1)
            pbar.close()
        return

    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    q: Queue = Queue(maxsize=args.queue_size)

    writer = Thread(target=writer_loop, args=(args.out_pattern, args.maxcount, q, total), daemon=True)
    writer.start()

    procs: List[Process] = []
    for idxs in chunk_indices(len(zips), n_workers):
        pz = [zips[i] for i in idxs]
        pc = [cohorts[i] for i in idxs]
        pp = [patient_ids[i] for i in idxs]
        proc = Process(target=worker_reader, args=(pz, pc, pp, args.coords_regex, q), daemon=True)
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    q.put(None)
    writer.join()

if __name__ == "__main__":
    main()
