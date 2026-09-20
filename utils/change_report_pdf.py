from decimal import Decimal, ROUND_HALF_UP
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

FONT = "STSong-Light"
FACTORS = {"m2": Decimal("1"), "ha": Decimal("0.0001"), "km2": Decimal("0.000001")}
DECIMALS = {"m2": 2, "ha": 4, "km2": 4}
LABELS = {"m2": "平方米", "ha": "公顷", "km2": "平方公里"}

def safe_text(value):
    text = "未记录" if value is None else str(value)
    result = []
    for char in text:
        try:
            char.encode("gb2312")
        except UnicodeEncodeError:
            char = "□"
        result.append(char)
    return "".join(result)

def format_area_m2(value, unit):
    if unit not in FACTORS:
        raise ValueError("unit 必须为 m2、ha 或 km2")
    if value is None:
        return "未记录"
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("面积必须为有限非负数")
    if number == 0:
        return "0"
    converted = number * FACTORS[unit]
    quantum = Decimal(1).scaleb(-DECIMALS[unit])
    if converted < quantum:
        return f"<{quantum:f}"
    return f"{converted.quantize(quantum, rounding=ROUND_HALF_UP):f}"

def register_font():
    if FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(FONT))


import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

def _table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),FONT),("GRID",(0,0),(-1,-1),.3,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
    return table

def build_pdf(report):
    register_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("report-body", parent=styles["BodyText"], fontName=FONT, fontSize=9, leading=13)
    heading = ParagraphStyle("report-heading", parent=body, fontSize=13, leading=18)
    title = ParagraphStyle("report-title", parent=body, fontSize=18, leading=24)
    p = lambda value, style=body: Paragraph(safe_text(value), style)
    story = [p(report["title"], title), p(f"结果 #{report['result_id']} · 检测时间：{report['detection_time']} · 报告生成时间：{report['generated_at']}"), Spacer(1,8), p("报告参数", heading)]
    rows = [[p("字段"),p("前期"),p("后期")]]
    for key,label in [("name","影像名称"),("capture_date","采集日期"),("satellite","卫星"),("resolution","分辨率"),("crs","源 CRS"),("size","源像素宽高")]:
        rows.append([p(label),p(report["before"][key]),p(report["after"][key])])
    rows.append([p("模型"),p(report["model"]),p(report["model"])])
    story.append(_table(rows,[35*mm,65*mm,65*mm]))
    for period,label in [("before","前期"),("after","后期")]:
        story += [PageBreak(),p(f"{label}影像与分类图", heading),p(f"{label}原始影像"),Image(io.BytesIO(report[period]["original_png"]),width=165*mm,height=70*mm),Spacer(1,5),p(f"{label}分类图"),Image(io.BytesIO(report[period]["classification_png"]),width=165*mm,height=70*mm)]
    story += [PageBreak(),p("面积与转移矩阵",heading),p(f"面积单位：{LABELS[report['unit']]}；单期面积统计整幅有效区域，矩阵统计共同有效区域。")]
    rows = [[p("类别"),p("前期"),p("后期")]]
    for i,item in enumerate(report["classes"]):
        rows.append([p(item["name"]),p(format_area_m2(report["before_area"][i],report["unit"])),p(format_area_m2(report["after_area"][i],report["unit"]))])
    story += [Spacer(1,6),_table(rows,[70*mm,55*mm,55*mm]),Spacer(1,8),p(f"共同有效区域面积：{format_area_m2(report['common_valid_area_m2'],report['unit'])} {LABELS[report['unit']]}")]
    rows = [[p("前期／后期")] + [p(item["name"]) for item in report["classes"]]]
    for i,item in enumerate(report["classes"]):
        rows.append([p(item["name"])] + [p(format_area_m2(value,report["unit"])) for value in report["matrix_m2"][i]])
    story.append(_table(rows,[25*mm]+[25*mm]*len(report["classes"])))
    output = io.BytesIO()
    SimpleDocTemplate(output,pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=16*mm,bottomMargin=16*mm).build(story)
    return output.getvalue()

