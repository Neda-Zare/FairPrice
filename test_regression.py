"""تست: دو روش جبر خطی باید جواب یکسان بدهند و با np.linalg.lstsq هم‌خوان باشند."""
import numpy as np
import pandas as pd
from regression import CarPriceModel, train_test_split

df = pd.read_csv("data/cars.csv")
print(f"ردیف‌های فایل: {len(df)}")
print("خانه‌های خالی هر ستون:", df.isna().sum().to_dict())

train, test = train_test_split(df)

ws = {}
for s in ("normal", "qr"):
    m = CarPriceModel(solver=s).fit(train)
    ws[s] = m.w_
    print(f"{s:7} ->", {k: round(float(v), 4) for k, v in m.score(test).items()})

print(f"ردیف‌های قابل استفاده برای آموزش: {m.n_used_}")

print(f"آگهی‌های با قیمت غیرواقعی که از آموزش حذف شدند: {m.n_trimmed_}")
Xa, ya = m.augmented_train_system()  # سیستم ریج روی دقیقاً همان ردیف‌هایی که مدل با آن‌ها آموزش دید
w_ref = np.linalg.lstsq(Xa, ya, rcond=None)[0]
for s, w in ws.items():
    err = np.max(np.abs(w - w_ref))
    assert err < 1e-6, (s, err)
    print(f"{s:7} max|w - lstsq| = {err:.2e}")

# تست قضاوت: ماشین با قیمت خیلی کم/زیاد
m = CarPriceModel(solver="qr").fit(df)
car = df.iloc[0].to_dict()
p = m.judge(car, 1)["predicted"]
assert "ارزان" in m.judge(car, p * 0.2)["verdict"]
assert "گران" in m.judge(car, p * 5)["verdict"]
assert "منصفانه" in m.judge(car, p)["verdict"]
print("همه‌ی تست‌ها موفق بود ✔")