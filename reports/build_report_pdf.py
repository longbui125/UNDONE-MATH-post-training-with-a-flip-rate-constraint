"""Render the editable Vietnamese research report as a self-contained PDF."""
from __future__ import annotations

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "bao_cao_grpo_rang_buoc_flip.md"
DESTINATION = HERE / "bao_cao_grpo_rang_buoc_flip.pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4
CONTENT_WIDTH = PAGE_WIDTH - 92
PDF_GLYPH_FALLBACKS = str.maketrans({
    "₁": "_1", "₄": "_4", "ₜ": "_t", "ᵢ": "_i", "ⱼ": "_j",
    "⁻": "^-", "⁶": "6", "⁸": "8",
})


def register_fonts() -> None:
    fonts = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("ArialVI", str(fonts / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("ArialVI-Bold", str(fonts / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("ArialVI-Italic", str(fonts / "ariali.ttf")))
    pdfmetrics.registerFontFamily(
        "ArialVI", normal="ArialVI", bold="ArialVI-Bold", italic="ArialVI-Italic"
    )


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    ink = colors.HexColor("#19304B")
    body = ParagraphStyle(
        "BodyVI", parent=base["Normal"], fontName="ArialVI", fontSize=9.4,
        leading=14.8, textColor=colors.HexColor("#263646"), spaceAfter=7,
        allowWidows=0, allowOrphans=0,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "TitleVI", parent=body, fontName="ArialVI-Bold", fontSize=21,
            leading=27, textColor=ink, spaceAfter=20, alignment=TA_LEFT,
        ),
        "h2": ParagraphStyle(
            "SectionVI", parent=body, fontName="ArialVI-Bold", fontSize=13,
            leading=18, textColor=ink, spaceBefore=15, spaceAfter=8,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "SubsectionVI", parent=body, fontName="ArialVI-Bold", fontSize=10.5,
            leading=15, textColor=colors.HexColor("#255C82"), spaceBefore=11,
            spaceAfter=6, keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "BulletVI", parent=body, leftIndent=15, firstLineIndent=-9,
            spaceAfter=4,
        ),
        "code": ParagraphStyle(
            "FormulaVI", parent=body, fontName="ArialVI", fontSize=8.6,
            leading=13.5, leftIndent=10, rightIndent=8, spaceBefore=2,
            spaceAfter=2, backColor=colors.HexColor("#F1F5F8"), borderPadding=6,
        ),
        "caption": ParagraphStyle(
            "CaptionVI", parent=body, fontName="ArialVI-Italic", fontSize=8.4,
            leading=12, textColor=colors.HexColor("#5D6A77"),
            alignment=TA_CENTER, spaceBefore=4, spaceAfter=11,
        ),
        "table_header": ParagraphStyle(
            "TableHeaderVI", parent=body, fontName="ArialVI-Bold", fontSize=7.5,
            leading=10, textColor=colors.white, spaceAfter=0,
        ),
        "table_cell": ParagraphStyle(
            "TableCellVI", parent=body, fontSize=7.5, leading=10,
            spaceAfter=0,
        ),
    }


TOKEN = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^]]+\]\([^)]+\))")


def inline_markdown(value: str) -> str:
    pieces = []
    for part in TOKEN.split(value):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            pieces.append(f"<b>{escape(part[2:-2])}</b>")
        elif part.startswith("*") and part.endswith("*"):
            pieces.append(f"<i>{escape(part[1:-1])}</i>")
        elif part.startswith("`") and part.endswith("`"):
            pieces.append(f"<font color='#255C82'>{escape(part[1:-1])}</font>")
        elif part.startswith("[") and "](" in part:
            label = part[1:part.index("](")]
            pieces.append(f"<font color='#255C82'>{escape(label)}</font>")
        else:
            pieces.append(escape(part))
    return "".join(pieces)


def table_flowable(lines: list[str], style: dict[str, ParagraphStyle]) -> LongTable:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append(cells)
    columns = len(rows[0])
    if columns == 7:
        widths = [73, 65, 57, 59, 66, 67, CONTENT_WIDTH - 387]
    elif columns == 5:
        widths = [CONTENT_WIDTH * .31] + [CONTENT_WIDTH * .1725] * 4
    elif columns == 4:
        widths = [CONTENT_WIDTH * .49] + [CONTENT_WIDTH * .17] * 3
    else:
        widths = [CONTENT_WIDTH / columns] * columns
    data = [
        [Paragraph(inline_markdown(cell),
                   style["table_header"] if index == 0 else style["table_cell"])
         for cell in row]
        for index, row in enumerate(rows)
    ]
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244A68")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F3F6F8")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#244A68")),
    ]))
    return table


def build_story(text: str) -> list:
    # Windows Arial lacks several Unicode math subscripts; keep formulas legible.
    text = text.translate(PDF_GLYPH_FALLBACKS)
    sty = styles()
    story = []
    paragraph = []
    table_lines = []
    code_lines = []
    in_code = False

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline_markdown(" ".join(paragraph)), sty["body"]))
            paragraph.clear()

    def flush_table() -> None:
        if table_lines:
            story.append(Spacer(1, 5))
            story.append(table_flowable(table_lines, sty))
            story.append(Spacer(1, 9))
            table_lines.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                story.append(KeepTogether([
                    Paragraph(escape(item).replace(" ", "&nbsp;"), sty["code"])
                    for item in code_lines
                ]))
                code_lines.clear()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(raw.rstrip())
            continue
        if line.startswith("|"):
            flush_paragraph()
            table_lines.append(line)
            continue
        flush_table()
        if not line:
            flush_paragraph()
            continue
        if line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(inline_markdown(line[2:]), sty["title"]))
            story.append(HRFlowable(width="100%", thickness=2,
                                    color=colors.HexColor("#2E7A9D")))
            story.append(Spacer(1, 14))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markdown(line[3:]), sty["h2"]))
        elif line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markdown(line[4:]), sty["h3"]))
        elif line.startswith("- ") or re.match(r"\d+\. ", line):
            flush_paragraph()
            body = re.sub(r"^\d+\.\s+", "", line[2:] if line.startswith("- ") else line)
            story.append(Paragraph("• " + inline_markdown(body), sty["bullet"]))
        elif line.startswith("!["):
            flush_paragraph()
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match:
                image_path = HERE / match.group(2)
                if image_path.is_file():
                    width, height = ImageReader(str(image_path)).getSize()
                    scale = min(CONTENT_WIDTH / width, 355 / height)
                    story.append(Spacer(1, 7))
                    story.append(Image(str(image_path), width=width * scale,
                                       height=height * scale, hAlign="CENTER"))
        elif line.startswith("*") and line.endswith("*"):
            flush_paragraph()
            story.append(Paragraph(inline_markdown(line), sty["caption"]))
        elif any(line.startswith(f"**{label}:**") for label in
                 ("Dự án", "Trạng thái", "Mô hình", "Dữ liệu", "Phạm vi kết quả")):
            flush_paragraph()
            story.append(Paragraph(inline_markdown(line), sty["body"]))
        else:
            paragraph.append(line.removesuffix("  "))
    flush_paragraph()
    flush_table()
    return story


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E0E6"))
    canvas.line(46, 39, PAGE_WIDTH - 46, 39)
    canvas.setFont("ArialVI", 7.7)
    canvas.setFillColor(colors.HexColor("#5C6C79"))
    canvas.drawString(46, 25, "PLMCO  |  Baseline GRPO có ràng buộc mềm  |  01/10/2026")
    canvas.drawRightString(PAGE_WIDTH - 46, 25, f"Trang {doc.page}")
    canvas.restoreState()


def main() -> None:
    register_fonts()
    document = SimpleDocTemplate(
        str(DESTINATION), pagesize=A4, leftMargin=46, rightMargin=46,
        topMargin=44, bottomMargin=55, title="Báo cáo GRPO với ràng buộc hạn chế đúng thành sai",
        author="Dự án PLMCO",
    )
    document.build(build_story(SOURCE.read_text(encoding="utf-8")),
                   onFirstPage=page_footer, onLaterPages=page_footer)
    print(DESTINATION)


if __name__ == "__main__":
    main()
