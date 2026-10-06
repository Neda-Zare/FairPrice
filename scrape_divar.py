"""
جمع‌آوری آگهی‌های خودرو از دیوار و ساخت data/cars.csv

اجرا (روی کامپیوتر خودتان، با اینترنت ایران):
    pip install requests
    python scrape_divar.py --city tehran --pages 40 --out data/cars.csv

نکات:
- ملاحظات: فقط برای پروژه‌ی درسی و با سرعت کم (پیش‌فرض ۲ ثانیه بین درخواست‌ها). حجم زیاد نزنید.
- ساختار API دیوار ممکن است عوض شود. پاسخ‌های خام در data/raw/ ذخیره می‌شوند؛
  اگر ستونی خالی ماند، یک فایل خام را باز کنید و تابع parse_post را مطابق آن اصلاح کنید.
- اگر شهر یا دسته‌بندی کار نکرد، آدرس واقعی را در مرورگر ببینید: divar.ir/s/<city>/car
"""
import argparse
import csv
import json
import os
import re
import time

import requests

API = "https://api.divar.ir/v8"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept": "application/json",
    "Content-Type": "application/json",
}
FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٫٬،", "0123456789.,,")

# نام فارسی برند در آگهی -> برچسب انگلیسی (نمونه؛ هر برند ناشناخته با همان نام فارسی می‌ماند)
COLUMNS = ["brand", "year", "mileage_km", "engine_cc", "automatic", "body_damage", "price", "token"]


def to_number(text):
    """تبدیل متن مثل '۱۲۰,۰۰۰ کیلومتر' یا '۴۵۰٬۰۰۰٬۰۰۰ تومان' به عدد"""
    if text is None:
        return None
    s = str(text).translate(FA_DIGITS).replace(",", "")
    m = re.search(r"\d+(\.\d+)?", s)
    return float(m.group()) if m else None


DEBUG = False


def request(session, method, url, retries=4, **kw):
    """درخواست با تلاش مجدد برای خطاهای 429 و 5xx"""
    for attempt in range(retries):
        r = session.request(method, url, timeout=30, **kw)
        if DEBUG:
            print(f"[debug] {method} {url} -> {r.status_code}  {r.text[:300]!r}")
        if r.status_code == 429 or r.status_code >= 500:
            wait = 20 * (attempt + 1)
            print(f"  وضعیت {r.status_code}؛ {wait} ثانیه صبر و تلاش مجدد ({attempt + 1}/{retries})...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def find_tokens(obj, out):
    """توکن آگهی‌ها را از ساختار پاسخ جستجو درمی‌آورد (ساختار دقیق ممکن است تغییر کند)"""
    if isinstance(obj, dict):
        d = obj.get("data")
        if obj.get("widget_type") == "POST_ROW" and isinstance(d, dict) and d.get("token"):
            out.append(d["token"])
        for v in obj.values():
            find_tokens(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_tokens(v, out)


def search_tokens(session, city, category, pages, delay):
    """گرفتن توکن آگهی‌ها با API وب دیوار: POST /v8/postlist/w/search (صفحه‌بندی با cursor)"""
    tokens, pagination = [], None
    for page in range(pages):
        payload = {
            "city_ids": [str(city)],
            "search_data": {"form_data": {"data": {"category": {"str": {"value": category}}}}},
        }
        if pagination:
            payload["pagination_data"] = pagination
        r = request(session, "POST", f"{API}/postlist/w/search", json=payload)
        data = r.json()
        found = []
        find_tokens(data, found)
        if not found:
            print(f"صفحه {page}: آگهی‌ای پیدا نشد؛ توقف.")
            break
        tokens += found
        pagination = (data.get("pagination") or {}).get("data")
        if not pagination:
            print("اطلاعات صفحه‌بندی در پاسخ نبود؛ توقف.")
            break
        print(f"صفحه {page}: {len(set(tokens))} آگهی تا اینجا")
        time.sleep(delay)
    return list(dict.fromkeys(tokens))


def flatten_fields(post):
    """همه‌ی جفت‌های (عنوان -> مقدار) آگهی را از ساختارهای مختلف پاسخ می‌کشد"""
    fields = {}

    def walk(o):
        if isinstance(o, dict):
            if "title" in o and "value" in o and isinstance(o["title"], str):
                fields[o["title"].strip()] = str(o["value"])
            # ردیف‌های ارزیابی (مثل «بدنه») مقدار را در descriptive_score می‌گذارند
            if "title" in o and "descriptive_score" in o and isinstance(o["title"], str):
                fields[o["title"].strip()] = str(o["descriptive_score"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(post)
    return fields


def find_price(post, fields):
    # دیوار قیمت را معمولاً به تومان می‌دهد؛ خروجی: میلیون تومان
    for key in ("قیمت", "قیمت پایه"):
        if key in fields:
            v = to_number(fields[key])
            if v:
                return v / 1_000_000
    w = ((post.get("webengage") or {}).get("price")
         or ((post.get("data") or {}).get("webengage") or {}).get("price"))
    if w:
        return float(w) / 1_000_000
    return None


def to_jalali_year(y):
    if y is None:
        return None
    return int(y) - 621 if y > 1700 else int(y)  # سال میلادی را تقریباً به شمسی تبدیل می‌کند


def clean_model(text):
    """'کیا اسپورتیج 2400cc' -> 'کیا اسپورتیج' (برند + مدل، بدون حجم موتور و تیپ)"""
    if not text:
        return None
    words = re.sub(r"\d+\s*cc", "", text.translate(FA_DIGITS), flags=re.I).split()
    return " ".join(words[:2]) or None


def parse_post(post, token):
    f = flatten_fields(post)
    seo = (post.get("seo") or {}).get("post_seo_schema") or {}
    model_text = f.get("برند و مدل") or f.get("برند و تیپ") or seo.get("model") or ""
    brand = clean_model((seo.get("brand") or {}).get("name")) or clean_model(model_text)

    year = to_jalali_year(to_number(f.get("مدل (سال تولید)") or f.get("سال ساخت") or f.get("مدل")))
    km_text = f.get("کارکرد", "")
    mileage = 0.0 if "صفر" in km_text else to_number(km_text)

    cc = re.search(r"(\d{3,4})\s*cc", model_text.translate(FA_DIGITS), flags=re.I)
    engine_cc = float(cc.group(1)) if cc else to_number(f.get("حجم موتور"))

    gearbox = f.get("گیربکس") or seo.get("vehicleTransmission") or ""
    automatic = 1 if "اتوماتیک" in gearbox else 0 if gearbox else None

    body = f.get("بدنه") or f.get("وضعیت بدنه") or ""
    if not body or "تعیین" in body:
        damage = None  # فروشنده وضعیت بدنه را مشخص نکرده
    elif "سالم" in body and "رنگ" not in body:
        damage = 0
    else:
        damage = 1 + body.count("،")  # تقریبی: تعداد قطعات ذکرشده

    return {
        "brand": brand, "year": year, "mileage_km": mileage,
        "engine_cc": engine_cc,
        "automatic": automatic, "body_damage": damage,
        "price": find_price(post, f), "token": token,
    }


def is_valid(row):
    """حذف آگهی‌های ناقص یا غیرعادی (قیمت توافقی/نمادین، سال یا کارکرد نامعقول)"""
    return bool(
        row["brand"] and row["year"] and row["price"]
        and row["mileage_km"] is not None
        and 1360 <= row["year"] <= 1410
        and 0 <= row["mileage_km"] <= 1_000_000
        and 30 <= row["price"] <= 100_000  # میلیون تومان
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="1", help="شناسه‌ی عددی شهر (تهران=1)")
    ap.add_argument("--category", default="cars", help="اسلاگ دسته‌بندی؛ اگر خالی برگشت 'light' یا 'car' را امتحان کنید")
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--out", default="data/cars.csv")
    ap.add_argument("--debug", action="store_true", help="چاپ وضعیت و بخشی از پاسخ هر درخواست")
    a = ap.parse_args()
    global DEBUG
    DEBUG = a.debug

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    s = requests.Session()
    s.headers.update(HEADERS)

    tokens = search_tokens(s, a.city, a.category, a.pages, a.delay)
    print(f"{len(tokens)} آگهی پیدا شد. دریافت جزئیات...")

    rows = []
    for i, t in enumerate(tokens, 1):
        try:
            post = request(s, "GET", f"{API}/posts-v2/web/{t}").json()
            if i <= 3:  # چند نمونه‌ی خام برای اشکال‌زدایی
                with open(f"data/raw/{t}.json", "w", encoding="utf-8") as fh:
                    json.dump(post, fh, ensure_ascii=False, indent=1)
            row = parse_post(post, t)
            if is_valid(row):
                rows.append(row)
        except Exception as e:  # آگهی خراب را رد کن
            print(f"  رد شد {t}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(tokens)} پردازش شد، {len(rows)} ردیف معتبر")
        time.sleep(a.delay)

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} ردیف در {a.out} ذخیره شد.")


if __name__ == "__main__":
    main()