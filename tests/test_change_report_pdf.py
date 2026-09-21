from io import BytesIO

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas

from utils.change_report_pdf import (
    FONT,
    PAGE_MARGIN_HORIZONTAL,
    _matrix_col_widths,
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
