"""
برنامه‌ی اصلی پروژه.

حالت تعاملی (بدون آرگومان؛ مثلاً با دکمه‌ی Run در PyCharm):
    python predict.py

آموزش مدل:
    python predict.py train --data data/cars.csv --solver qr

پیشنهاد قیمت:
    python predict.py price --brand "پژو 206" --year 1395 --mileage 160000

قضاوت قیمت (ارزان / منصفانه / گران):
    python predict.py judge --brand "پژو 206" --year 1395 --mileage 160000 --price 1100
"""
import argparse
import os
import sys
import pandas as pd
from regression import CarPriceModel, train_test_split, SOLVERS, NUMERIC

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")
DATA_PATH = os.path.join(HERE, "data", "cars.csv")


def ask(prompt, cast=str, default=None):
    """سؤال از کاربر؛ ورودی خالی = مقدار پیش‌فرض"""
    while True:
        s = input(prompt).strip()
        if not s and default is not None:
            return default
        try:
            return cast(s)
        except ValueError:
            print("  عدد معتبر وارد کنید.")


def get_model():
    """مدل ذخیره‌شده را برمی‌گرداند؛ اگر نبود یا مربوط به نسخه‌ی قدیمی برنامه بود، دوباره آموزش می‌دهد."""
    if os.path.exists(MODEL_PATH):
        try:
            m = CarPriceModel.load(MODEL_PATH)
            if getattr(m, "numeric_cols_", None) == NUMERIC:
                return m
            print("مدل ذخیره‌شده مربوط به نسخه‌ی قدیمی است؛ دوباره آموزش می‌دهم ...")
        except Exception:
            print("مدل ذخیره‌شده خوانده نشد؛ دوباره آموزش می‌دهم ...")
    else:
        print("مدلی ذخیره نشده؛ در حال آموزش روی data/cars.csv ...")
    m = CarPriceModel("qr").fit(pd.read_csv(DATA_PATH))
    m.save(MODEL_PATH)
    return m


def interactive():
    """حالت تعاملی: اگر مدل ذخیره نشده باشد خودش آموزش می‌دهد و بعد آگهی‌ها را یکی‌یکی می‌سنجد."""
    m = get_model()
    top = m.train_["brand"].value_counts().head(25).index.tolist()
    print("\nبرندها/مدل‌های پرتکرار در داده:")
    for i, b in enumerate(top, 1):
        print(f"  {i:>2}. {b}")
    print("(شماره یا نام را وارد کنید؛ هر نام دیگری هم قبول است. برای خروج، نام را خالی بگذارید.)\n")
    while True:
        b = input("برند و مدل: ").strip()
        if not b:
            break
        if b.isdigit() and 1 <= int(b) <= len(top):
            b = top[int(b) - 1]
        year = ask("سال ساخت (شمسی، مثلاً 1395): ", int)
        km = ask("کارکرد (کیلومتر، برای صفر 0): ", int)
        price = ask("قیمت آگهی به میلیون تومان (خالی = فقط پیشنهاد قیمت): ", float, 0.0)
        row = {"brand": b, "year": year, "mileage_km": km}
        r = m.judge(row, price if price > 0 else 1.0)
        print(f"\n  قیمت پیشنهادی مدل: {r['predicted']:,.0f} میلیون تومان")
        print(f"  بازه‌ی استاندارد: {r['range'][0]:,.0f} تا {r['range'][1]:,.0f}")
        if price > 0:
            print(f"  قیمت شما ({price:,.0f}): {r['verdict']}")
        print()


def car_row(a):
    return {"brand": a.brand, "year": a.year, "mileage_km": a.mileage}


def main():
    if len(sys.argv) == 1:
        interactive()
        return
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", default=DATA_PATH)
    t.add_argument("--solver", default="qr", choices=SOLVERS)
    t.add_argument("--no-log", action="store_true", help="رگرسیون روی خود قیمت (نه لگاریتم آن)")

    for name in ("price", "judge"):
        c = sub.add_parser(name)
        c.add_argument("--brand", required=True, help="مثلاً \"پژو 206\" (نام برند و مدل، همان‌طور که در داده آمده)")
        c.add_argument("--year", type=int, required=True, help="سال ساخت شمسی")
        c.add_argument("--mileage", type=int, required=True, help="کارکرد (کیلومتر)")
        if name == "judge":
            c.add_argument("--price", type=float, required=True, help="قیمت (میلیون تومان)")

    a = p.parse_args()

    if a.cmd == "train":
        df = pd.read_csv(a.data)
        train, test = train_test_split(df)
        m = CarPriceModel(solver=a.solver, log_target=not a.no_log).fit(train)
        print(f"روش حل: {a.solver} | ردیف‌های فایل: {len(df)} | آموزش: {m.n_used_} ردیف معتبر | تست: {len(m.prepare(test))} ردیف معتبر")
        print(f"آگهی‌های با قیمت غیرواقعی حذف‌شده از آموزش: {m.n_trimmed_}")
        print("ضرایب مدل (ویژگی‌های عددی استانداردشده):")
        print(f"  ضریب جریمه‌ی ریج λ = {m.lambda_}")
        names = m.feature_names
        for n, w in list(zip(names, m.w_))[:m.n_free_]:
            print(f"  {n:<22}{w:+.4f}")
        eff = sorted(list(zip(names, m.w_))[m.n_free_:], key=lambda t: -abs(t[1]))[:8]
        print("  قوی‌ترین اثرهای برند/مدل روی لگاریتم قیمت:")
        for n, w in eff:
            print(f"    {n:<28}{w:+.3f}")
        print("ارزیابی روی داده‌ی تست:", {k: round(float(v), 3) for k, v in m.score(test).items()})
        m = CarPriceModel(solver=a.solver, log_target=not a.no_log).fit(df)  # مدل نهایی روی کل داده
        m.save(MODEL_PATH)
        print(f"مدل در {MODEL_PATH} ذخیره شد")
        return

    m = get_model()
    row = car_row(a)
    if a.cmd == "price":
        price = float(m.predict(pd.DataFrame([row]))[0])
        print(f"قیمت پیشنهادی: {price:,.0f} میلیون تومان")
    else:
        r = m.judge(row, a.price)
        print(f"قیمت پیشنهادی مدل: {r['predicted']:,.0f} میلیون تومان")
        print(f"بازه‌ی استاندارد: {r['range'][0]:,.0f} تا {r['range'][1]:,.0f}")
        print(f"قیمت شما ({a.price:,.0f}): {r['verdict']}")


if __name__ == "__main__":
    main()