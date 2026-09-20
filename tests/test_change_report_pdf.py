from io import BytesIO

from reportlab.pdfgen.canvas import Canvas

from utils.change_report_pdf import FONT, format_area_m2, register_font, safe_text

def test_area_formatting_boundaries():
    assert format_area_m2(0, 'ha') == '0'
    assert format_area_m2(0.5, 'ha') == '<0.0001'
    assert format_area_m2(10000, 'ha') == '1.0000'
    assert format_area_m2(None, 'ha') == '未记录'

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
