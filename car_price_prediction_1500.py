"""
Car Price Prediction - 1500 cars dataset
Libraries: pandas, numpy, matplotlib ONLY (no scikit-learn)
Needs: car_data_1500.csv in the same folder as this script.
Run:   python car_price_prediction_1500.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_YEAR = 2026
folder = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. LOAD ----------
df = pd.read_csv(os.path.join(folder, "car_data_1500.csv"))
print("Shape:", df.shape)
print(df.head(), "\n")
print("Missing values:", int(df.isnull().sum().sum()), "\n")

# ---------- 2. CLEAN + FEATURES ----------
df = df.drop_duplicates().dropna()
df["car_age"] = CURRENT_YEAR - df["year"]
df = df.drop(columns="year")
counts = df["brand"].value_counts()
df["brand"] = df["brand"].where(df["brand"].map(counts) >= 15, "Other")   # group rare brands

y = df["selling_price_lakh"].values
X = pd.get_dummies(df.drop(columns="selling_price_lakh"),
                   columns=["brand", "fuel", "seller_type", "transmission"],
                   drop_first=True, dtype=float)
feature_names = list(X.columns)
X = X.values.astype(float)

# ---------- 3. SPLIT 80/20 + SCALE ----------
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
cut = int(0.8 * len(X))
tr, te = idx[:cut], idx[cut:]
mean, std = X[tr].mean(axis=0), X[tr].std(axis=0)
std[std == 0] = 1
Xs = (X - mean) / std

# ---------- 4. LINEAR REGRESSION FROM SCRATCH ----------
def fit(X, y):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)      # normal equation
    return beta

def predict(X, beta):
    return np.hstack([np.ones((len(X), 1)), X]) @ beta

def scores(a, p):
    r2 = 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
    return r2, np.mean(np.abs(a - p)), np.sqrt(np.mean((a - p) ** 2))

beta_a = fit(Xs[tr], y[tr])                              # Model A: plain price
pred_a = np.clip(predict(Xs[te], beta_a), 0.1, None)
beta_b = fit(Xs[tr], np.log(y[tr]))                      # Model B: log price
pred_b = np.exp(predict(Xs[te], beta_b))

# ---------- 5. RESULTS ----------
print("---- Test-set results (price in lakhs) ----")
print(f"{'Model':<28}{'R2':>8}{'MAE':>10}{'RMSE':>10}")
for name, p in [("A) Plain price", pred_a), ("B) Log price (improved)", pred_b)]:
    r2, mae, rmse = scores(y[te], p)
    print(f"{name:<28}{r2:>8.3f}{mae:>10.3f}{rmse:>10.3f}")
print(f"\nModel B typical error: about {np.median(np.abs(y[te]-pred_b)/y[te])*100:.0f}% of the price\n")

coef = pd.Series(beta_b[1:], index=feature_names).sort_values()
print("Factors that RAISE price:\n", coef.tail(5).round(3), "\n")
print("Factors that LOWER price:\n", coef.head(5).round(3), "\n")

# ---------- 6. PLOTS ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for a, p, t in [(ax[0], pred_a, "A) Plain price"), (ax[1], pred_b, "B) Log price")]:
    a.scatter(y[te], p, alpha=0.5, s=14)
    a.plot([0, 60], [0, 60], "r--"); a.set_title(t); a.set_xlabel("Actual price (lakhs)")
ax[0].set_ylabel("Predicted price (lakhs)"); ax[0].set_xlim(0, 60); ax[0].set_ylim(0, 60)
plt.tight_layout(); plt.savefig("actual_vs_predicted_1500.png", dpi=120); plt.close()

plt.figure(figsize=(7, 7)); coef.plot(kind="barh")
plt.title("What affects price? (log model)"); plt.tight_layout()
plt.savefig("feature_coefficients_1500.png", dpi=120); plt.close()

# ---------- 7. PREDICT A NEW CAR ----------
def predict_price(brand, age, km, owner, engine_cc, power_bhp,
                  fuel="Petrol", seller="Individual", transmission="Manual"):
    row = dict.fromkeys(feature_names, 0.0)
    row.update({"car_age": age, "km_driven": km, "owner": owner,
                "engine_cc": engine_cc, "power_bhp": power_bhp})
    for key in (f"brand_{brand}", f"fuel_{fuel}", f"seller_type_{seller}", f"transmission_{transmission}"):
        if key in row:
            row[key] = 1.0
    x = (np.array([[row[f] for f in feature_names]]) - mean) / std
    return float(np.exp(predict(x, beta_b))[0])

print("Example predictions (data is from ~2020-2022, so cars are 4+ years old):")
print(f"  Maruti, 8 yrs, 60,000 km, 1197cc, 82bhp, petrol -> {predict_price('Maruti', 8, 60000, 1, 1197, 82):.2f} lakhs")
print(f"  Hyundai, 7 yrs, 70,000 km, 1493cc, 110bhp, diesel -> {predict_price('Hyundai', 7, 70000, 1, 1493, 110, fuel='Diesel', seller='Dealer'):.2f} lakhs")
