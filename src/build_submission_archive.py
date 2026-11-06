"""生成投稿归档包：submission/ids_experiment_submission_v1.zip

收录：
- src/*.py 全部源代码（不含 __pycache__）
- results/*.json、*.csv、*.md、*.log（不含 proba/ 大体积概率文件）
- paper/manuscript_v0.3.md、supplementary_S1.md、figures/*.png、SUBMISSION_CHECKLIST.md
- submission/README.md（复制进包内）

概率文件（results/proba/*.npy，约 3.6 GB）不进包：可由存档权重与脚本重生成
（见 README），或审稿人要求时另行提供。

用法： /d/Anaconda/python src/build_submission_archive.py
"""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submission"
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / "ids_experiment_submission_v1.zip"

INCLUDE_PATTERNS = [
    ("src", "*.py"),
    ("results", "*.json"),
    ("results", "*.csv"),
    ("results", "*.md"),
    ("results", "*.log"),
    ("paper", "manuscript_v0.3.md"),
    ("paper", "manuscript_v0.3.docx"),
    ("paper", "supplementary_S1.md"),
    ("paper", "supplementary_S1.pdf"),
    ("paper", "SUBMISSION_CHECKLIST.md"),
    ("paper/figures", "*.png"),
    ("submission", "README.md"),
    ("submission", "zenodo_metadata.md"),
]

def main():
    if OUT.exists():
        OUT.unlink()
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for sub, pat in INCLUDE_PATTERNS:
            d = ROOT / sub
            if not d.exists():
                print("missing:", d)
                continue
            for p in sorted(d.glob(pat)):
                z.write(p, Path(sub) / p.name)
                n += 1
    size = OUT.stat().st_size / 1024 / 1024
    print(f"archive: {OUT}")
    print(f"files: {n}   size: {size:.1f} MB")

if __name__ == "__main__":
    main()
