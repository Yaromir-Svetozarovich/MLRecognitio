"""DOCX renderer for answer blanks with ArUco markers.

Renders a printable A4 answer sheet (DOCX) from layout + content JSON:
  - 4 ArUco markers (DICT_4X4_50) anchored at page corners as floating images
  - Header (author, group, ФИО, variant, course)
  - UPPERCASE instruction
  - Table of questions with numbered rows and bordered answer cells

Version is encoded by the BL marker ID = content["version_marker_id"].
"""

from __future__ import annotations

import cv2
import os
import tempfile
from pathlib import Path

from docx import Document
from docx.shared import Mm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from app.template.generator import resolve_marker_ids

EMU_PER_MM = 36000
TWIPS_PER_MM = 56.7


def mm_to_px(mm, dpi=300):
    """Convert millimetres to pixels at the given DPI (default 300)."""
    return int(round(mm / 25.4 * dpi))


def generate_marker(marker_id, size_mm, quiet_zone_mm, dpi=300):
    """Generate a single ArUco marker PNG with quiet zone border.

    Args:
        marker_id: ArUco dictionary ID.
        size_mm: physical marker size in millimetres.
        quiet_zone_mm: quiet zone width in millimetres on each side.
        dpi: render resolution (default 300).

    Returns:
        numpy array (BGR) of the marker with white quiet zone border.
    """
    size_px = mm_to_px(size_mm, dpi)
    quiet_px = mm_to_px(quiet_zone_mm, dpi)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.aruco.generateImageMarker(d, marker_id, size_px)
    return cv2.copyMakeBorder(
        img, quiet_px, quiet_px, quiet_px, quiet_px,
        cv2.BORDER_CONSTANT, value=255,
    )


def validate_marker(img, expected_id):
    """Self-check that a generated marker image decodes to the expected ID.

    Args:
        img: marker image (BGR or grayscale).
        expected_id: the ArUco ID we expect to decode.

    Returns:
        True if the marker is detected and matches expected_id.
    """
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(img)
    return ids is not None and expected_id in ids.flatten()


def add_anchored_picture(paragraph, image_path, x_mm, y_mm, w_mm, h_mm):
    """Insert a picture as a floating anchored image at absolute page position.

    Converts the inline picture created by run.add_picture() into a
    wp:anchor element positioned relative to the page at (x_mm, y_mm).
    This is what places markers at exact millimetre coordinates on the page.

    Args:
        paragraph: the docx paragraph to attach the anchor to.
        image_path: path to the marker PNG file.
        x_mm, y_mm: top-left position on the page in millimetres.
        w_mm, h_mm: width and height of the anchored image in millimetres.

    Raises:
        RuntimeError: if the inline element could not be created.
    """
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
    """Apply single black borders to all four edges of a table cell."""
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
    """Remove all borders from a table cell (set val=nil)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_b = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'nil')
        tc_b.append(el)
    tc_pr.append(tc_b)


def set_cell_width_mm(cell, width_mm):
    """Set the width of a table cell in millimetres (converted to dxa/twips).

    Updates the existing ``w:tcW`` element instead of appending a duplicate,
    which Word would otherwise ignore.
    """
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn('w:tcW'))
    if tc_w is None:
        tc_w = OxmlElement('w:tcW')
        tc_pr.append(tc_w)
    tc_w.set(qn('w:w'), str(int(width_mm * TWIPS_PER_MM)))
    tc_w.set(qn('w:type'), 'dxa')


def set_table_grid(table, widths_mm):
    """Set fixed table grid columns and total table width in millimetres.

    Without updating ``w:tblGrid``, Word/LibreOffice keep three equal columns
    regardless of per-cell widths, which breaks ROI alignment.
    """
    tbl = table._tbl
    grid = tbl.tblGrid
    for grid_col in list(grid.findall(qn('w:gridCol'))):
        grid.remove(grid_col)
    for width_mm in widths_mm:
        grid_col = OxmlElement('w:gridCol')
        grid_col.set(qn('w:w'), str(int(width_mm * TWIPS_PER_MM)))
        grid.append(grid_col)

    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.find(qn('w:tblW'))
    if tbl_w is None:
        tbl_w = OxmlElement('w:tblW')
        tbl_pr.append(tbl_w)
    tbl_w.set(qn('w:w'), str(int(sum(widths_mm) * TWIPS_PER_MM)))
    tbl_w.set(qn('w:type'), 'dxa')

    tbl_layout = tbl_pr.find(qn('w:tblLayout'))
    if tbl_layout is None:
        tbl_layout = OxmlElement('w:tblLayout')
        cell_mar = tbl_pr.find(qn('w:tblCellMar'))
        look = tbl_pr.find(qn('w:tblLook'))
        anchor = cell_mar if cell_mar is not None else look
        if anchor is not None:
            anchor.addprevious(tbl_layout)
        else:
            tbl_pr.append(tbl_layout)
    tbl_layout.set(qn('w:type'), 'fixed')


def set_row_height_mm(row, height_mm, rule='atLeast'):
    """Set the height of a table row in millimetres.

    Args:
        row: docx table row object.
        height_mm: desired row height in millimetres.
        rule: 'atLeast' (minimum) or 'exact'.
    """
    tr_pr = row._tr.get_or_add_trPr()
    tr_h = OxmlElement('w:trHeight')
    tr_h.set(qn('w:val'), str(int(height_mm * TWIPS_PER_MM)))
    tr_h.set(qn('w:hRule'), rule)
    tr_pr.append(tr_h)


def set_table_cell_margins(table, left_mm=0.0, right_mm=0.0, top_mm=0.0, bottom_mm=0.0):
    """Set fixed cell margins for the whole table.

    Zero left/right margins make the table border and grid columns align
    exactly with page millimetres, so ROI coordinates stay in sync.
    """
    tbl_pr = table._tbl.tblPr
    old = tbl_pr.find(qn('w:tblCellMar'))
    if old is not None:
        tbl_pr.remove(old)
    margins = OxmlElement('w:tblCellMar')
    for edge, value in (
        ('top', top_mm),
        ('left', left_mm),
        ('bottom', bottom_mm),
        ('right', right_mm),
    ):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:w'), str(int(value * TWIPS_PER_MM)))
        el.set(qn('w:type'), 'dxa')
        margins.append(el)
    look = tbl_pr.find(qn('w:tblLook'))
    if look is not None:
        look.addprevious(margins)
    else:
        tbl_pr.append(margins)


def set_table_position(table, left_mm, top_mm):
    """Position a table absolutely on the page (floating table).

    Uses ``w:tblpPr`` with page-relative anchors, so the table starts at an
    exact (left, top) offset regardless of the text that precedes it. This is
    what keeps the answer-cell geometry in sync with ``roi_fields``.
    """
    tbl_pr = table._tbl.tblPr
    tblp_pr = OxmlElement('w:tblpPr')
    tblp_pr.set(qn('w:horzAnchor'), 'page')
    tblp_pr.set(qn('w:vertAnchor'), 'page')
    tblp_pr.set(qn('w:tblpX'), str(int(left_mm * TWIPS_PER_MM)))
    tblp_pr.set(qn('w:tblpY'), str(int(top_mm * TWIPS_PER_MM)))
    # tblpPr must be the first child of tblPr in the OOXML schema.
    tbl_pr.insert(0, tblp_pr)


def add_paragraph(doc, text, size=11, bold=False, align=None,
                  space_before=0, space_after=2):
    """Add a paragraph with the given text formatting to the document.

    Args:
        doc: the Document object.
        text: paragraph text (empty string creates a spacer).
        size: font size in points.
        bold: whether the run is bold.
        align: WD_ALIGN_PARAGRAPH value or None.
        space_before / space_after: paragraph spacing in points.
    """
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
    """High-level renderer wrapper: render(layout, content, output_path) -> Path."""

    def render(self, layout: dict, content: dict, output_path: str | Path) -> Path:
        return generate_blank(layout, content, str(output_path))


def generate_blank(layout, content, output_path):
    """Generate a complete answer-blank DOCX from layout + content JSON.

    Steps:
      1. Configure page size and margins from layout.page / layout.margins.
      2. Draw 4 anchored ArUco markers at page corners (TL, TR, BL, BR).
      3. Add title, header info, instruction paragraph.
      4. Add a questions table: [number | question text | bordered answer cell].
      5. Save DOCX to output_path.

    The BL marker ID is taken from content["version_marker_id"], which
    encodes the template version into the physical sheet for later
    normalization validation. TL/TR/BR IDs come from layout.markers.ids.

    Args:
        layout: parsed template_layout.json.
        content: parsed content_vN.json.
        output_path: path where the DOCX will be saved.

    Returns:
        Path to the saved DOCX file.
    """
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

    marker_ids = resolve_marker_ids(layout, content)

    tmpdir = tempfile.mkdtemp(prefix="aruco_blank_")

    # Zero-height paragraph used as anchor container for floating marker images
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

    # Title
    add_paragraph(doc, 'БЛАНК ОТВЕТОВ', size=14, bold=True,
                  align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)

    # Header info
    h = content["header"]
    add_paragraph(doc, f"Разработал: {h.get('author', '')}     "
                       f"Группа: {h.get('group', '')}", size=10, space_after=0)
    add_paragraph(doc, "ФИО учащегося: " + "_" * 40, size=10, space_after=0)
    add_paragraph(doc, f"Вариант: {h.get('variant', '')}     "
                       f"Класс/курс: {h.get('course', '')}", size=10, space_after=0)

    # Instruction (uppercase)
    add_paragraph(doc, '', size=2)
    add_paragraph(doc, 'ИНСТРУКЦИЯ: ' + content["instruction"].upper(),
                  size=9, bold=True, space_after=4)

    # Questions table: [number | question text | bordered answer cell].
    # The table is floating at page-relative (left, top), so its answer-cell
    # coordinates match build_roi_fields() exactly.
    qs = content["questions"]
    roi_cfg = layout.get("roi") or {}
    table_top_mm = float(roi_cfg.get("table_top_mm", 78))
    row_height_mm = float(roi_cfg.get("row_height_mm", 7))
    table_left_mm = float(mg["left_mm"])

    table = doc.add_table(rows=len(qs), cols=3)
    table.autofit = False
    set_table_position(table, table_left_mm, table_top_mm)

    widths = layout["question_columns_mm"]  # e.g. [12, 112, 56]
    set_table_grid(table, widths)
    set_table_cell_margins(table)

    for idx, q in enumerate(qs):
        row = table.rows[idx]
        set_row_height_mm(row, row_height_mm, rule='exact')
        cells = row.cells

        # Column 0: question number
        cells[0].text = ''
        p = cells[0].paragraphs[0]
        r = p.add_run(f"{q['num']}.")
        r.font.size = Pt(9); r.bold = True; r.font.name = 'Arial'
        clear_cell_borders(cells[0])
        set_cell_width_mm(cells[0], widths[0])

        # Column 1: question text
        cells[1].text = ''
        p = cells[1].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(q['text'])
        r.font.size = Pt(8.5); r.font.name = 'Arial'
        clear_cell_borders(cells[1])
        set_cell_width_mm(cells[1], widths[1])

        # Column 2: empty bordered answer cell (student writes here)
        cells[2].text = ''
        set_cell_borders(cells[2], sz=12)
        set_cell_width_mm(cells[2], widths[2])

    doc.save(output_path)
    return Path(output_path)