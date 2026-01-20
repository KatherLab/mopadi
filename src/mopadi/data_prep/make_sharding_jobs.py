#!/usr/bin/env python3
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")

COHORTS = """
TCGA-BLCA  TCGA-CESC  TCGA-CRC   TCGA-ESCA  TCGA-HNSC  TCGA-KIRC  TCGA-LGG   TCGA-LUAD  TCGA-MESO  TCGA-PAAD  TCGA-PRAD  TCGA-SKCM  TCGA-TGCT  TCGA-THYM  TCGA-UCS
TCGA-BRCA  TCGA-CHOL  TCGA-DLBC  TCGA-GBM   TCGA-KICH  TCGA-KIRP  TCGA-LIHC  TCGA-LUSC  TCGA-OV    TCGA-PCPG  TCGA-SARC  TCGA-STAD  TCGA-THCA  TCGA-UCEC  UCL-GBM
""".strip()

# Adjust these if your paths or resources change
ZIP_BASE = f'{ws_path}/cache/pngs'
SHARD_BASE = f'{ws_path}/shards-new'
WORKERS = 8
CPUS_PER_TASK = 8  # keep in sync with WORKERS
GPU_SPEC = 'gpu:1'
TIME = '8:00:00'
MEM_PER_CPU_MB = 2000

OUT_DIR = Path(f'{ws_path}/mopadi/jobs/sharding')

TEMPLATE = """#!/bin/sh

#SBATCH --job-name="shard-{cohort}"
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --gres={gpu}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --tasks-per-node=1
#SBATCH --output=outs/sharding/shard-{cohort}-output_%j.out

python src/mopadi/data_prep/convert_to_shards.py \\
  --workers {workers} \\
  --zip_glob "{zip_base}/{cohort}/*.zip" \\
  --out_pattern "{shard_base}/{cohort}/{cohort}-%06d.tar" \\
  --cohort "{cohort}" \\
  --maxcount 8000
"""

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohorts = [c for c in COHORTS.replace('\n', ' ').split() if c.strip()]
    for cohort in cohorts:
        script_text = TEMPLATE.format(
            cohort=cohort,
            time=TIME,
            gpu=GPU_SPEC,
            cpus=CPUS_PER_TASK,
            mem_per_cpu=MEM_PER_CPU_MB,
            workers=WORKERS,
            zip_base=ZIP_BASE.rstrip('/'),
            shard_base=SHARD_BASE.rstrip('/'),
        )
        script_path = OUT_DIR / f"shard-{cohort}.slurm"
        script_path.write_text(script_text)
        # make it executable
        os.chmod(script_path, 0o755)

    print(f"Generated {len(cohorts)} SLURM files in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
