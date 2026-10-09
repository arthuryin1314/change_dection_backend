from io import BytesIO

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas

from utils.change_report_pdf import (
    FONT,
    PAGE_MARGIN_HORIZONTAL,
    _matrix_col_widths,
    build_pdf,
    format_area_m2,
    register_font,
    safe_text,
)


@pytest.mark.parametrize('unit, value, expected', [
    ('m2', 0, '0'),
    ('m2', 0.001, '<0.01'),
    ('m2', 12345678901.23, '12345678901.23'),
    ('ha', 0, '0'),
    ('ha', 0.5, '<0.0001'),
    ('ha', 12345678901.0, '1234567.8901'),
    ('km2', 0, '0'),
    ('km2', 0.5, '<0.0001'),
    ('km2', 12345678901.0, '12345.6789'),
])
def test_area_formatting_boundaries(unit, value, expected):
    assert format_area_m2(value, unit) == expected


def test_area_formatting_keeps_large_values_in_fixed_decimal_notation():
    assert format_area_m2(100, 'm2') == '100'
    assert format_area_m2(1e10, 'm2') == '10000000000'
    assert format_area_m2(10000, 'ha') == '1'
    assert format_area_m2(None, 'ha') == '未记录'


def test_matrix_columns_fit_page_and_longest_values_fit_cells():
    register_font()
    widths = _matrix_col_widths(6)
    doc_width = A4[0] - 2 * PAGE_MARGIN_HORIZONTAL
    assert sum(widths) <= doc_width
    for value in ('12345678901.23', '1234567.8901', '12345.6789'):
        text_width = pdfmetrics.stringWidth(value, FONT, 9)
        assert text_width + 12 <= widths[1]


def test_cid_text_falls_back_for_unmapped_characters():
    assert '□' in safe_text('影像🛰️.tif')


def test_cid_font_smoke_generates_pdf():
    register_font()
    output = BytesIO()
    canvas = Canvas(output)
    canvas.setFont(FONT, 12)
    canvas.drawString(72, 760, safe_text('影像🛰️.tif 生僻字𠮷 繁體'))
    canvas.save()
    assert output.getvalue().startswith(b'%PDF')


def test_pdf_contains_persisted_transition_page_and_legacy_note():
    image = BytesIO()
    from PIL import Image as PILImage
    PILImage.new("RGB", (1, 1), "white").save(image, format="PNG")
    image_data = image.getvalue()
    report = {
        "title": "变化检测结果报告",
        "detection_time": "2026-09-20 08:00:00",
        "generated_at": "2026-09-20 08:01:00",
        "unit": "ha",
        "model": "model",
        "classes": [{"id": i, "name": name} for i, name in enumerate(("其他／背景", "水系", "林地", "道路", "种植土地", "房屋建筑"))],
        "before": {**{key: "未记录" for key in ("name", "capture_date", "satellite", "resolution", "crs", "size")}, "original_png": image_data, "classification_png": image_data},
        "after": {**{key: "未记录" for key in ("name", "capture_date", "satellite", "resolution", "crs", "size")}, "original_png": image_data, "classification_png": image_data},
        "before_area": [0] * 6,
        "after_area": [0] * 6,
        "common_valid_area_m2": 0,
        "matrix_m2": [[0] * 6 for _ in range(6)],
        "legacy_transition_note": "未保存多值转移结果；以下仅显示历史矩阵。",
        "transition_png": None,
        "transition_legend": None,
        "transition_snapshot": None,
    }
    output = build_pdf(report)
    assert output.startswith(b"%PDF")
