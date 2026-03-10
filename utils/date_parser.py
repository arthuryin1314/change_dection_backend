import re
from datetime import date, datetime


def parse_capture_date(value: str) -> date:
    """Parse capture_date from common frontend formats to a date object.

    Supported examples:
    - 2026-03-01
    - 2026-03-01T00:00:00
    - Sun Mar 01 2026 00:00:00 GMT+0800 (China Standard Time)
    """
    raw = (value or "").strip()
    if not raw:
        raise ValueError("capture_date 不能为空")

    # 1) Strict date format: YYYY-MM-DD
    try:
        return date.fromisoformat(raw)
    except ValueError:
        pass

    # 2) ISO datetime (with/without timezone)
    try:
        normalized_iso = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized_iso).date()
    except ValueError:
        pass

    # 3) JS Date string: remove trailing timezone text in parentheses
    #    e.g. "Sun Mar 01 2026 00:00:00 GMT+0800 (中国标准时间)"
    normalized = re.sub(r"\s*\(.*\)\s*$", "", raw)

    js_formats = (
        "%a %b %d %Y %H:%M:%S GMT%z",
        "%a %b %d %Y %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S GMT%z",
    )
    for fmt in js_formats:
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue

    raise ValueError("capture_date 格式不正确，支持 YYYY-MM-DD 或 JS Date 字符串")
