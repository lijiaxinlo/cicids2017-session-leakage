"""对 pandoc 生成的 docx 做排版后处理：表宽、字号、表头重复、图片限高。"""
import sys
from docx import Document
from docx.shared import Pt, Cm, Emu
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

path = sys.argv[1]
doc = Document(path)
sec = doc.sections[0]
USABLE = Emu(sec.page_width - sec.left_margin - sec.right_margin)
MAX_IMG_H = Cm(15.0)

# ---------- 图片限高 ----------
for sh in doc.inline_shapes:
    if sh.height > MAX_IMG_H:
        ratio = MAX_IMG_H / sh.height
        sh.height = int(sh.height * ratio)
        sh.width = int(sh.width * ratio)

# ---------- 表格 ----------
def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, spec in kwargs.items():
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        for k, v in spec.items():
            el.set(qn(f"w:{k}"), str(v))

for ti, table in enumerate(doc.tables, 1):
    ncol = len(table.columns)
    size = 8.0 if ncol <= 6 else (7.5 if ncol <= 8 else 7.0)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tblPr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)

    # 列宽按内容长度加权
    weights = []
    for c in range(ncol):
        lens = []
        for r in table.rows:
            try:
                lens.append(len(r.cells[c].text.strip()))
            except IndexError:
                pass
        weights.append(max(6, min(34, max(lens) if lens else 8)))
    total = sum(weights)
    widths = [Emu(int(USABLE * w / total)) for w in weights]

    for row in table.rows:
        # 行不跨页
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
        for ci, cell in enumerate(row.cells):
            if ci < len(widths):
                cell.width = widths[ci]
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.name = "Times New Roman"

    # 表头：加粗、跨页重复、上下横线
    hdr = table.rows[0]
    trPr = hdr._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    trPr.append(th)
    for cell in hdr.cells:
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(size)
        set_cell_border(cell, top={"val": "single", "sz": 8, "color": "000000"},
                        bottom={"val": "single", "sz": 6, "color": "000000"})
    for cell in table.rows[-1].cells:
        set_cell_border(cell, bottom={"val": "single", "sz": 8, "color": "000000"})

doc.save(path)
print(f"postprocessed: {path} | tables={len(doc.tables)} | images={len(doc.inline_shapes)}")
