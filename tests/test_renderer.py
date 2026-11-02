import json
import zipfile
from pathlib import Path

import cv2
import pytest

from app.template.renderer import generate_blank


def load_fixtures():
    root = Path(__file__).resolve().parents[1]
    layout = json.loads((root / "template_layout.json").read_text(encoding="utf-8"))
    content = json.loads((root / "content_v1.json").read_text(encoding="utf-8"))
    return layout, content


def test_render_produces_docx(tmp_path):
    layout, content = load_fixtures()
    out = tmp_path / "blank.docx"
    generate_blank(layout, content, str(out))
    assert out.exists()
    # reopen sanity
    from docx import Document
    Document(str(out))

    # 4 images embedded
    with zipfile.ZipFile(out) as z:
        media = [n for n in z.namelist() if n.startswith("word/media/")]
    assert len(media) == 4

    # markers decodable
    import os
    d = tmp_path / "imgs"
    d.mkdir()
    with zipfile.ZipFile(out) as z:
        for name in media:
            (d / Path(name).name).write_bytes(z.read(name))
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )
    ids_found = []
    for f in sorted(d.iterdir()):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        _, ids, _ = det.detectMarkers(img)
        assert ids is not None
        ids_found.append(int(ids.flatten()[0]))
    assert sorted(ids_found) == [0, 1, 2, content["version_marker_id"]]