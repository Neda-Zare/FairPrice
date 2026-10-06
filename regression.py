"""
هسته‌ی رگرسیون خطی برای پروژه‌ی جبر خطی (قیمت‌گذاری خودرو)

مدل:   y = X w + e
که X ماتریس ویژگی‌ها (با ستون ثابت 1 برای عرض از مبدأ) و y لگاریتم قیمت (یا خود قیمت) است.
دو روش جبر خطی برای حل مسئله‌ی کمترین مربعات  min ||X w - y||^2  پیاده شده:

1) معادله‌ی نرمال:  (X^T X) w = X^T y
2) تجزیه‌ی QR:      X = Q R  ->  R w = Q^T y   (حل با جایگزینی پسرو)
"""
import numpy as np
import pandas as pd

NUMERIC = ["year", "mileage_km"]  # ویژگی‌های عددی؛ علاوه بر این‌ها برند/مدل هم در مدل هست
TARGET = "price"  # میلیون تومان
RIDGE_LAMBDA = 1.0  # شدت جریمه‌ی ریج روی ضریب‌های برند (بزرگ‌تر = برندهای کم‌آگهی بیشتر به میانگین نزدیک می‌شوند)
CV_FOLDS = 5  # فقط برای برآورد خطای مدل روی آگهی‌های دیده‌نشده (σ در تشخیص ارزان/گران)
TRIM_MADS = 3.5  # آگهی‌هایی که خطایشان از این‌قدر MAD بیشتر باشد (قیمت غیرواقعی) از آموزش حذف می‌شوند


def make_of(brand):
    """سازنده = اولین کلمه‌ی نام (مثلاً «پژو 206» -> «پژو»)"""
    return str(brand).split()[0]


def basic_filter(df):
    """تبدیل ستون‌ها به عدد و حذف ردیف‌های نامعتبر (قیمت توافقی/صفر، سال یا کارکرد نامعقول، بدون برند)"""
    df = df.copy()
    for c in NUMERIC + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = df["brand"].notna() & df[TARGET].gt(0)
    ok &= df["year"].between(1340, 1410)
    ok &= df["mileage_km"].ge(0)
    return df[ok]


# ---------------------------------------------------------------- حل‌کننده‌ها
def solve_normal(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


def solve_qr(X, y):
    Q, R = np.linalg.qr(X)  # تجزیه‌ی QR کاهش‌یافته
    return back_substitution(R, Q.T @ y)


def back_substitution(R, b):
    """حل دستگاه بالامثلثی R w = b"""
    n = R.shape[0]
    w = np.zeros(n)
    for i in range(n - 1, -1, -1):
        w[i] = (b[i] - R[i, i + 1:] @ w[i + 1:]) / R[i, i]
    return w


SOLVERS = {"normal": solve_normal, "qr": solve_qr}


# ---------------------------------------------------------------- مدل
class CarPriceModel:
    """
    رگرسیون ریج (کمترین مربعات با جریمه‌ی تیخونوف):
        min ||X w - y||^2 + lambda * ||w_brand||^2
    فقط ضریب‌های برند/مدل جریمه می‌شوند (نه عرض از مبدأ و ویژگی‌های عددی). معادل است با حل کمترین مربعات
    روی ماتریس افزوده‌ی [X ; sqrt(lambda) P] و بردار [y ; 0] که P ردیف‌های انتخاب ستون‌های برند است.
    برندهای پرتکرار ضریب بزرگ می‌گیرند و برندهای کم‌آگهی به‌طور خودکار به سمت صفر (میانگین) کشیده می‌شوند.
    """

    def __init__(self, solver="qr", log_target=True):
        self.solver = solver
        self.log_target = log_target

    # ---- سطوح برند: «سازنده» (کلمه‌ی اول) و «مدل» (نام کامل، فقط اگر حداقل دو آگهی داشته باشد)
    def _fit_levels(self, df):
        self.makes_ = sorted(df["brand"].map(make_of).unique())
        counts = df["brand"].value_counts()
        self.models_ = sorted(b for b, c in counts.items() if c >= 2 and len(str(b).split()) > 1)

    # ---- ساخت ماتریس ویژگی
    def _design(self, df, fit=False):
        cols = [df[c].astype(float).to_numpy() for c in NUMERIC]
        mk = df["brand"].map(make_of)
        cols += [(mk == m).astype(float).to_numpy() for m in self.makes_]
        cols += [(df["brand"] == m).astype(float).to_numpy() for m in self.models_]
        A = np.column_stack(cols)
        k = len(NUMERIC)
        if fit:  # استانداردسازی فقط روی ستون‌های عددی (برای پایداری عددی)
            self.mu_ = A[:, :k].mean(axis=0)
            self.sd_ = A[:, :k].std(axis=0)
        A[:, :k] = (A[:, :k] - self.mu_) / self.sd_
        return np.column_stack([np.ones(len(A)), A])

    @property
    def n_free_(self):
        """تعداد ستون‌های بدون جریمه: عرض از مبدأ + ویژگی‌های عددی"""
        return 1 + len(NUMERIC)

    @property
    def feature_names(self):
        return (["intercept"] + NUMERIC
                + [f"make={m}" for m in self.makes_] + [f"model={m}" for m in self.models_])

    def _augment(self, X, y, lam):
        """ماتریس افزوده‌ی ریج: ردیف‌های sqrt(lam)*I برای ستون‌های برند به X اضافه می‌شود"""
        p = X.shape[1] - self.n_free_
        P = np.hstack([np.zeros((p, self.n_free_)), np.sqrt(lam) * np.eye(p)])
        return np.vstack([X, P]), np.concatenate([y, np.zeros(p)])

    def _ridge(self, X, y, lam):
        Xa, ya = self._augment(X, y, lam)
        return SOLVERS[self.solver](Xa, ya)

    def augmented_train_system(self):
        """سیستم افزوده‌ی آموزش (برای تست با lstsq)"""
        X = self._design(self.train_)
        y = self.train_[TARGET].to_numpy(float)
        return self._augment(X, np.log(y) if self.log_target else y, self.lambda_)

    # ---- پاک‌سازی (یکسان برای آموزش، تست و امتیازدهی)
    def prepare(self, df):
        """فقط ردیف‌های کامل و معتبر را نگه می‌دارد؛ هر جا مدل با ردیف ناقص کار کند NaN تولید می‌شود."""
        df = basic_filter(df)
        need = NUMERIC + ["brand", TARGET]
        return df.dropna(subset=need).reset_index(drop=True)

    def _target(self, df):
        y = df[TARGET].to_numpy(float)
        return np.log(y) if self.log_target else y

    def _cv_rmse(self, X, y):
        """خطای مدل روی آگهی‌های دیده‌نشده: داده را ۵ بخش می‌کنیم، هر بار روی ۴ بخش آموزش و روی بخش پنجم می‌سنجیم"""
        folds = np.random.default_rng(0).permutation(len(y)) % CV_FOLDS
        sq = 0.0
        for f in range(CV_FOLDS):
            tr, va = folds != f, folds == f
            w = self._ridge(X[tr], y[tr], self.lambda_)
            sq += np.sum((y[va] - X[va] @ w) ** 2)
        return float(np.sqrt(sq / len(y)))

    def _fit_once(self, df):
        """ساخت X و آموزش ریج با λ ثابت. خروجی: X, y, w"""
        self._fit_levels(df)
        self.lambda_ = RIDGE_LAMBDA
        X = self._design(df, fit=True)
        y = self._target(df)
        self.cv_rmse_ = self._cv_rmse(X, y)
        return X, y, self._ridge(X, y, self.lambda_)

    # ---- آموزش
    def fit(self, df, trim=True):
        df = basic_filter(df)
        df = df.dropna(subset=NUMERIC + ["brand", TARGET]).reset_index(drop=True)
        if len(df) < 30:
            raise ValueError(f"فقط {len(df)} ردیف کامل و معتبر باقی ماند؛ داده برای آموزش کافی نیست.")
        self.numeric_cols_ = list(NUMERIC)  # برای تشخیص مدل ذخیره‌شده‌ی قدیمی هنگام بارگذاری
        self.n_trimmed_ = 0
        X, y, w = self._fit_once(df)
        if trim:  # حذف آگهی‌های با قیمت غیرواقعی (مثلاً کرولا ۸۴ میلیون) و آموزش دوباره
            res = y - X @ w
            mad = np.median(np.abs(res - np.median(res))) * 1.4826
            keep = np.abs(res - np.median(res)) <= TRIM_MADS * mad
            self.n_trimmed_ = int((~keep).sum())
            if self.n_trimmed_:
                df = df[keep].reset_index(drop=True)
                X, y, w = self._fit_once(df)
        self.w_ = w
        self.train_ = df  # ردیف‌هایی که واقعاً در آموزش استفاده شدند
        self.n_used_ = len(df)
        self.resid_std_ = float(self.cv_rmse_)  # σ از اعتبارسنجی متقابل (صادقانه‌تر از خطای آموزش)
        return self

    def predict(self, df):
        yhat = self._design(df) @ self.w_
        return np.exp(yhat) if self.log_target else yhat

    # ---- ارزیابی
    def score(self, df):
        df = self.prepare(df)
        y = df[TARGET].to_numpy(float)
        p = self.predict(df)
        ape = np.abs(y - p) / y
        ly, lp = np.log(y), np.log(p)
        return {
            "R2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
            "R2_log": 1 - np.sum((ly - lp) ** 2) / np.sum((ly - ly.mean()) ** 2),
            "MAE": float(np.mean(np.abs(y - p))),
            "MAPE_%": float(np.mean(ape) * 100),
            "MedAPE_%": float(np.median(ape) * 100),
        }

    # ---- تشخیص ارزان / گران
    def judge(self, row, price, k=0.5):
        """
        قیمت پیشنهادی مدل را با قیمت واقعی مقایسه می‌کند.
        بازه‌ی «استاندارد» = قیمت مدل × exp(± k·σ)  (σ = خطای اعتبارسنجی متقابل مدل)
        """
        pred = float(self.predict(pd.DataFrame([row]))[0])
        if self.log_target:
            lo, hi = pred * np.exp(-k * self.resid_std_), pred * np.exp(k * self.resid_std_)
        else:
            lo, hi = pred - k * self.resid_std_, pred + k * self.resid_std_
        if price < lo:
            verdict = "پایین‌تر از حد استاندارد (ارزان)"
        elif price > hi:
            verdict = "بالاتر از حد استاندارد (گران)"
        else:
            verdict = "برابر حد استاندارد (منصفانه)"
        return {"predicted": pred, "range": (lo, hi), "verdict": verdict}

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def train_test_split(df, test_frac=0.2, seed=0):
    idx = np.random.default_rng(seed).permutation(len(df))
    n_test = int(len(df) * test_frac)
    return df.iloc[idx[n_test:]], df.iloc[idx[:n_test]]