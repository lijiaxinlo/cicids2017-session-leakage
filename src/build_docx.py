"""把 paper/manuscript_v0.3.md 构建成 MDPI 风格 docx（pandoc + 后处理）。

用法： python src/build_docx.py [源md] [输出docx]
依赖： pandoc 在 PATH 中；python-docx（系统 Python 已装）
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAPER, "manuscript_v0.3.md")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PAPER, "manuscript_v0.3.docx")

build = os.path.join(PAPER, ".build")
os.makedirs(build, exist_ok=True)

text = open(SRC, encoding="utf-8").read()
lines = text.split("\n")

# 标题行形如 "# Title"，去掉井号后作为 pandoc metadata
title = re.sub(r"^#+\s*", "", lines[0].strip())
body = [l for l in lines[1:] if not l.startswith("**Manuscript version:**") and l.strip() != "---"]

# 截断内部工作笔记：Appendix A 标注了 "not part of the submission"，不能出现在投稿件里
cut = None
for i, l in enumerate(body):
    if l.startswith("## ") and "not part of the submission" in l:
        cut = i
        break
if cut is not None:
    print(f"strip internal appendix at body line {cut}")
    body = body[:cut]

# 章节标题整体上提一级，使 ## -> Heading 1、### -> Heading 2
out = []
for l in body:
    if l.startswith("### "):
        l = "## " + l[4:]
    elif l.startswith("## "):
        l = "# " + l[3:]
    if l.strip() == "" and out and out[-1].strip() == "":
        continue
    out.append(l)

body_path = os.path.join(build, "ms_body.md")
open(body_path, "w", encoding="utf-8").write("\n".join(out).lstrip("\n"))

ref_doc = os.path.join(PAPER, "mdpi_reference.docx")
if not os.path.exists(ref_doc):
    raise SystemExit("缺少参考模板 paper/mdpi_reference.docx，先运行 src/make_mdpi_ref.py")

cmd = [
    "pandoc", body_path, "-o", OUT,
    "--reference-doc", ref_doc,
    "--resource-path", PAPER,
    "--metadata", f"title={title}",
]
print("pandoc ->", OUT)
subprocess.run(cmd, check=True, cwd=PAPER)

# 排版后处理
subprocess.run([sys.executable, os.path.join(ROOT, "src", "postprocess_docx.py"), OUT], check=True)
print("done:", OUT)
