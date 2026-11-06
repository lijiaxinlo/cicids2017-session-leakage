"""把 paper/supplementary_S1.md 导出为 A4 PDF（pandoc -> HTML -> Edge headless 打印）。

排版沿用既有 S1 PDF：A4、页边距 16mm/17mm、正文 9.5pt、表格与代码块细线风格。

用法： /d/Anaconda/python src/build_s1_pdf.py
依赖： pandoc（Anaconda 自带）、Microsoft Edge（headless --print-to-pdf）
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SRC = PAPER / "supplementary_S1.md"
BUILD = PAPER / ".build"
BUILD.mkdir(exist_ok=True)
HTML = BUILD / "s1.html"
CSS = BUILD / "s1.css"
OUT = PAPER / "supplementary_S1.pdf"

CSS_TEXT = """
@page { size: A4; margin: 16mm 17mm; }
body { font-family: "Times New Roman", "Songti SC", serif; font-size: 9.5pt;
       line-height: 1.35; color: #111; }
h1 { font-size: 13pt; margin: 0 0 6pt 0; }
h2 { font-size: 11pt; margin: 10pt 0 4pt 0; border-bottom: 0.5pt solid #999;
     padding-bottom: 2pt; }
h3 { font-size: 10pt; margin: 8pt 0 3pt 0; }
p { margin: 0 0 5pt 0; text-align: justify; }
table { table-layout: fixed; border-collapse: collapse; font-size: 8.5pt; margin: 6pt 0; width: 100%; }
th, td { border: 0.4pt solid #888; padding: 2pt 4pt; text-align: left;
         vertical-align: top; }
th { background: #eee; }
code { font-family: Consolas, monospace; font-size: 8.5pt; }
pre { background: #f5f5f5; border: 0.4pt solid #ccc; padding: 5pt;
      font-size: 8.2pt; line-height: 1.25; white-space: pre-wrap; }
blockquote { margin: 4pt 0 4pt 10pt; padding-left: 6pt;
             border-left: 2pt solid #bbb; color: #333; }
a { color: #0645ad; text-decoration: none; word-break: break-all; }
"""

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def main():
    CSS.write_text(CSS_TEXT, encoding="utf-8")
    subprocess.run(
        ["pandoc", str(SRC), "-o", str(HTML), "--standalone",
         "--css", "s1.css"],
        check=True,
    )
    edge = next((p for p in EDGE_CANDIDATES if os.path.exists(p)), None)
    if edge is None:
        sys.exit("找不到 Edge/Chrome，无法打印 PDF")
    if OUT.exists():
        OUT.unlink()
    # 独立 user-data-dir 必需：否则 Edge 会挂到已运行实例上、子进程不退出（表现为超时）
    profile = BUILD / "edgeprofile"
    subprocess.run(
        [edge, "--headless", "--disable-gpu", "--no-sandbox",
         f"--user-data-dir={profile}", "--no-pdf-header-footer",
         f"--print-to-pdf={OUT}", HTML.as_uri()],
        check=True, timeout=120,
    )
    print(f"pdf -> {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
