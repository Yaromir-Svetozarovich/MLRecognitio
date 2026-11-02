"""DOCX renderer for answer blanks with ArUco markers."""

from __future__ import annotations

import cv2
import os
import tempfile
from pathlib import Path

from docx import Document
from docx.shared import Mm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

EMU_PER_MM = 36000
TWIPS_PER_MM = 56.7


def mm_to_px(mm, dpi=300):
    return int(round(mm / 25.4 * dpi))


def generate_marker(marker_id, size_mm, quiet_zone_mm, dpi=300):
    size_px = mm_to_px(size_mm, dpi)
    quiet_px = mm_to_px(quiet_zone_mm, dpi)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.aruco.generateImageMarker(d, marker_id, size_px)
    return cv2.copyMakeBorder(
        img, quiet_px, quiet_px, quiet_px, quiet_px,
        cv2.BORDER_CONSTANT, value=255,
    )


def validate_marker(img, expected_id):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(img)
    return ids is not None and expected_id in ids.flatten()


def add_anchored_picture(paragraph, image_path, x_mm, y_mm, w_mm, h_mm):
    run = paragraph.add_run()
    run.add_picture(image_path, width=Mm(w_mm), height=Mm(h_mm))
    inline = run._element.find(qn('w:drawing'))
    if inline is not None:
        inline = inline.find(qn('wp:inline'))
    if inline is None:
        raise RuntimeError('Inline element not created')

    extent = inline.find(qn('wp:extent'))
    docPr = inline.find(qn('wp:docPr'))
    graphic = inline.find(qn('a:graphic'))

    anchor = OxmlElement('wp:anchor')
    for k, v in [
        ('distT', '0'), ('distB', '0'), ('distL', '0'), ('distR', '0'),
        ('simplePos', '0'), ('relativeHeight', '251659264'),
        ('behindDoc', '0'), ('locked', '0'),
        ('layoutInCell', '1'), ('allowOverlap', '1'),
    ]:
        anchor.set(k, v)

    sp = OxmlElement('wp:simplePos')
    sp.set('x', '0'); sp.set('y', '0')
    anchor.append(sp)

    ph = OxmlElement('wp:positionH')
    ph.set('relativeFrom', 'page')
    off = OxmlElement('wp:posOffset')
    off.text = str(int(x_mm * EMU_PER_MM))
    ph.append(off); anchor.append(ph)

    pv = OxmlElement('wp:positionV')
    pv.set('relativeFrom', 'page')
    off = OxmlElement('wp:posOffset')
    off.text = str(int(y_mm * EMU_PER_MM))
    pv.append(off); anchor.append(pv)

    anchor.append(extent)
    anchor.append(docPr)
    anchor.append(graphic)
    inline.getparent().replace(inline, anchor)


def set_cell_borders(cell, sz=12, color='000000'):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_b = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), str(sz))
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), color)
        tc_b.append(el)
    tc_pr.append(tc_b)


def clear_cell_borders(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_b = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'nil')
        tc_b.append(el)
    tc_pr.append(tc_b)


def set_cell_width_mm(cell, width_mm):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = OxmlElement('w:tcW')
    tc_w.set(qn('w:w'), str(int(width_mm * TWIPS_PER_MM)))
    tc_w.set(qn('w:type'), 'dxa')
    tc_pr.append(tc_w)


def set_row_height_mm(row, height_mm, rule='atLeast'):
    tr_pr = row._tr.get_or_add_trPr()
    tr_h = OxmlElement('w:trHeight')
    tr_h.set(qn('w:val'), str(int(height_mm * TWIPS_PER_MM)))
    tr_h.set(qn('w:hRule'), rule)
    tr_pr.append(tr_h)


def add_paragraph(doc, text, size=11, bold=False, align=None,
                  space_before=0, space_after=2):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.font.name = 'Arial'
        run.bold = bold
    if align is not None:
        p.alignment = align
    return p



class TemplateRenderer:
    """Renders an answer blank DOCX from layout + content."""

    def render(self, layout: dict, content: dict, output_path: str | Path) -> Path:
        return generate_blank(layout, content, str(output_path))


def generate_blank(layout, content, output_path):
    page = layout["page"]
    mg = layout["margins"]
    mc = layout["markers"]
    total_mm = mc["size_mm"] + 2 * mc["quiet_zone_mm"]

    doc = Document()
    s = doc.sections[0]
    s.page_width = Mm(page["width_mm"])
    s.page_height = Mm(page["height_mm"])
    s.top_margin = Mm(mg["top_mm"])
    s.bottom_margin = Mm(mg["bottom_mm"])
    s.left_margin = Mm(mg["left_mm"])
    s.right_margin = Mm(mg["right_mm"])

    marker_ids = {
        "TL": mc["ids"]["TL"],
        "TR": mc["ids"]["TR"],
        "BL": mc["ids"]["BL"],
        "BR": content["version_marker_id"],
    }

    tmpdir = tempfile.mkdtemp(prefix="aruco_blank_")

    p_markers = doc.add_paragraph()
    p_markers.paragraph_format.space_before = Pt(0)
    p_markers.paragraph_format.space_after = Pt(0)
    r0 = p_markers.add_run()
    r0.font.size = Pt(1)

    for corner, pos in mc["positions"].items():
        mid = marker_ids[corner]
        img = generate_marker(mid, mc["size_mm"], mc["quiet_zone_mm"])
        if not validate_marker(img, mid):
            raise RuntimeError(f"Marker self-check failed: ID={mid} at {corner}")
        path = os.path.join(tmpdir, f"m_{corner}_{mid}.png")
        cv2.imwrite(path, img)
        add_anchored_picture(
            p_markers, path,
            x_mm=pos["x_mm"], y_mm=pos["y_mm"],
            w_mm=total_mm, h_mm=total_mm,
        )

    add_paragraph(doc, 'БЛАНК ОТВЕТОВ', size=14, bold=True,
                  align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)

    h = content["header"]
    add_paragraph(doc, f"Разработал: {h.get('author', '')}     "
                       f"Группа: {h.get('group', '')}", size=10, space_after=0)
    add_paragraph(doc, "ФИО учащегося: " + "_" * 40, size=10, space_after=0)
    add_paragraph(doc, f"Вариант: {h.get('variant', '')}     "
                       f"Класс/курс: {h.get('course', '')}", size=10, space_after=0)

    add_paragraph(doc, '', size=2)
    add_paragraph(doc, 'ИНСТРУКЦИЯ: ' + content["instruction"].upper(),
                  size=9, bold=True, space_after=4)

    qs = content["questions"]
    table = doc.add_table(rows=len(qs), cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    widths = layout["question_columns_mm"]

    for idx, q in enumerate(qs):
        row = table.rows[idx]
        set_row_height_mm(row, 7, rule='atLeast')
        cells = row.cells

        cells[0].text = ''
        p = cells[0].paragraphs[0]
        r = p.add_run(f"{q['num']}.")
        r.font.size = Pt(9); r.bold = True; r.font.name = 'Arial'
        clear_cell_borders(cells[0])
        set_cell_width_mm(cells[0], widths[0])

        cells[1].text = ''
        p = cells[1].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(q['text'])
        r.font.size = Pt(8.5); r.font.name = 'Arial'
        clear_cell_borders(cells[1])
        set_cell_width_mm(cells[1], widths[1])

        cells[2].text = ''
        set_cell_borders(cells[2], sz=12)
        set_cell_width_mm(cells[2], widths[2])

    doc.save(output_path)
    return Path(output_path)