"""生成 MDPI 风格的 pandoc 参考模板（A4、Times New Roman 10pt、两端对齐）。"""
import sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn

src, dst = sys.argv[1], sys.argv[2]
doc = Document(src)

# 页面：A4，上下 2.5 cm，左右 1.8 cm（MDPI Word 模板）
for s in doc.sections:
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.top_margin = s.bottom_margin = Cm(2.5)
    s.left_margin = s.right_margin = Cm(1.8)


def set_font(style, size=None, bold=None, italic=None, color=None, name="Times New Roman"):
    f = style.font
    f.name = name
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = color
    # 同时设置 East Asian 字体，避免 Word 回退
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)


def para(style, before=0, after=0, spacing=1.0, align=None, keep_next=None):
    pf = style.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = spacing
    if align is not None:
        pf.alignment = align
    if keep_next is not None:
        pf.keep_with_next = keep_next


S = doc.styles

if "Normal" in [s.name for s in S]:
    st = S["Normal"]
    set_font(st, 10)
    para(st, 0, 0, 1.0, WD_ALIGN_PARAGRAPH.JUSTIFY)

for name, size, bold, italic, before, after, align in [
    ("Body Text", 10, False, False, 0, 6, WD_ALIGN_PARAGRAPH.JUSTIFY),
    ("First Paragraph", 10, False, False, 0, 6, WD_ALIGN_PARAGRAPH.JUSTIFY),
    ("Compact", 10, False, False, 0, 2, WD_ALIGN_PARAGRAPH.JUSTIFY),
    ("Title", 18, True, False, 0, 10, WD_ALIGN_PARAGRAPH.LEFT),
    ("Subtitle", 12, False, False, 0, 8, WD_ALIGN_PARAGRAPH.LEFT),
    ("Heading 1", 12, True, False, 14, 6, WD_ALIGN_PARAGRAPH.LEFT),
    ("Heading 2", 11, True, False, 10, 4, WD_ALIGN_PARAGRAPH.LEFT),
    ("Heading 3", 10, True, True, 8, 4, WD_ALIGN_PARAGRAPH.LEFT),
    ("Image Caption", 9, False, False, 2, 10, WD_ALIGN_PARAGRAPH.JUSTIFY),
    ("Table Caption", 9, False, False, 8, 2, WD_ALIGN_PARAGRAPH.JUSTIFY),
    ("Caption", 9, False, False, 2, 8, WD_ALIGN_PARAGRAPH.JUSTIFY),
    ("Bibliography", 9, False, False, 0, 3, WD_ALIGN_PARAGRAPH.LEFT),
    ("Block Text", 9, False, False, 0, 6, WD_ALIGN_PARAGRAPH.JUSTIFY),
]:
    if name in [s.name for s in S]:
        st = S[name]
        set_font(st, size, bold, italic, RGBColor(0, 0, 0))
        para(st, before, after, 1.0, align, keep_next=True if name.startswith("Heading") else None)

# 表格内文字 8 pt
for nm in ("Table",):
    if nm in [s.name for s in S]:
        set_font(S[nm], 8)
        para(S[nm], 0, 0, 1.0, WD_ALIGN_PARAGRAPH.LEFT)

doc.save(dst)
print("saved", dst)
