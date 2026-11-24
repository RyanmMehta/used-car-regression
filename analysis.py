"""
Used car prices: should the dealer buy this car?

Reproduces every number in the slides. Run from this folder:
    python analysis.py

Steps (matching the assignment):
  1. Split the 240 sold cars 70/30 into training and test sets (fixed seed).
  2. Fit a linear regression of sale price on the features (training cars only).
     Drop any feature that does not earn its place, compare adjusted R^2.
  3. Compare baseline (training average) vs v1 on the test cars: R^2 and RMSE.
  4. Apply the buy/pass rule (and the baseline rule) to the 40 auction lots.
Outputs go to outputs/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 42            # fixed so the split (and every number) is the same each run
TEST_SHARE = 0.30
PREP_COST = 1_500    # inspection, reconditioning, detailing, lot time per car
P_CUTOFF = 0.05      # a feature "earns its place" if its p-value is below this

ALL_FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]


def load_cars(path):
    """Read a car file and turn Accident yes/no into 1/0 so the regression can use it."""
    df = pd.read_csv(path)
    df["Accident"] = (df["Accident"].str.strip().str.lower() == "yes").astype(int)
    return df


def fit_ols(train, features):
    X = sm.add_constant(train[features])
    return sm.OLS(train["Sale_Price"], X).fit()


def predict(model, features, cars):
    X = sm.add_constant(cars[features], has_constant="add")
    return model.predict(X)


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


# ---------------------------------------------------------------- 1. Split
sales = load_cars(DATA / "used_car_sales.csv")
train, test = train_test_split(sales, test_size=TEST_SHARE, random_state=SEED)

print("=" * 70)
print("1. DATA SPLIT")
print(f"   {len(sales)} cars total -> {len(train)} training, {len(test)} test (seed {SEED})")

# ---------------------------------------------------------------- 2. Fit
full = fit_ols(train, ALL_FEATURES)
print("=" * 70)
print("2a. FULL MODEL (all five features), training cars")
print(full.summary())

# Drop features whose p-value is above the cutoff, one at a time (worst first),
# refitting each time, until every remaining feature earns its place.
kept = list(ALL_FEATURES)
dropped = []
model = full
while True:
    pvals = model.pvalues.drop("const")
    worst = pvals.idxmax()
    if pvals[worst] < P_CUTOFF:
        break
    kept.remove(worst)
    dropped.append((worst, float(pvals[worst])))
    model = fit_ols(train, kept)

reduced = model
print("=" * 70)
if dropped:
    print("2b. REDUCED MODEL (dropped: " + ", ".join(f"{f} p={p:.3f}" for f, p in dropped) + ")")
    print(reduced.summary())
else:
    print("2b. Every feature has p < 0.05, nothing to drop.")

print("=" * 70)
print("2c. ADJUSTED R^2 COMPARISON (training cars)")
print(f"   Full model    ({len(ALL_FEATURES)} features): {full.rsquared_adj:.4f}")
print(f"   Reduced model ({len(kept)} features): {reduced.rsquared_adj:.4f}")

# v1 = the reduced model if it is at least as good on adjusted R^2 (simpler wins ties),
# otherwise the full model.
if dropped and reduced.rsquared_adj >= full.rsquared_adj - 0.001:
    v1, V1_FEATURES = reduced, kept
    v1_note = "reduced model (simpler, adjusted R^2 as good or better)"
else:
    v1, V1_FEATURES = full, ALL_FEATURES
    v1_note = "full model"
print(f"   -> v1 = {v1_note}: {V1_FEATURES}")

coef_table = pd.DataFrame({
    "coefficient": v1.params,
    "p_value": v1.pvalues,
    "ci_low_95": v1.conf_int()[0],
    "ci_high_95": v1.conf_int()[1],
}).round(4)
coef_table.to_csv(OUT / "v1_coefficients.csv")

# ---------------------------------------------------------------- 3. Test-set fit
baseline_price = float(train["Sale_Price"].mean())
test_actual = test["Sale_Price"]
test_base = np.full(len(test), baseline_price)
test_v1 = predict(v1, V1_FEATURES, test)

fit_results = pd.DataFrame({
    "model": ["Baseline (training average)", "v1 (regression)"],
    "test_R2": [r2_score(test_actual, test_base), r2_score(test_actual, test_v1)],
    "test_RMSE_dollars": [rmse(test_actual, test_base), rmse(test_actual, test_v1)],
}).round(4)
fit_results.to_csv(OUT / "test_fit_baseline_vs_v1.csv", index=False)

print("=" * 70)
print("3. FIT ON THE 72 TEST CARS (never seen by the model)")
print(f"   Baseline guess for every car = training average = ${baseline_price:,.0f}")
print(fit_results.to_string(index=False))

# ---------------------------------------------------------------- 4. Decision rule


def dealer_rule(car, auction_price):
    """v1 rule: buy if auction price + prep cost is below the predicted sale price."""
    car_df = pd.DataFrame([car]) if not isinstance(car, pd.DataFrame) else car
    predicted = float(predict(v1, V1_FEATURES, car_df).iloc[0])
    return "buy" if auction_price + PREP_COST < predicted else "pass"


def baseline_rule(car, auction_price):
    """Baseline rule: buy if auction price + prep cost is below the training average."""
    return "buy" if auction_price + PREP_COST < baseline_price else "pass"


lots = load_cars(DATA / "auction_lots.csv")
lots["Predicted_Sale_v1"] = predict(v1, V1_FEATURES, lots).round(0)
lots["Profit_if_bought"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP_COST
lots["Baseline_decision"] = [
    baseline_rule(row[ALL_FEATURES], row["Auction_Price"]) for _, row in lots.iterrows()
]
lots["v1_decision"] = [
    dealer_rule(row[ALL_FEATURES], row["Auction_Price"]) for _, row in lots.iterrows()
]
lots.to_csv(OUT / "auction_decisions.csv", index=False)


def score(decision_col):
    bought = lots[lots[decision_col] == "buy"]
    return {
        "cars_bought": len(bought),
        "money_losing_cars_bought": int((bought["Profit_if_bought"] < 0).sum()),
        "total_profit_dollars": int(bought["Profit_if_bought"].sum()),
    }


decision_results = pd.DataFrame(
    [{"rule": "Baseline", **score("Baseline_decision")},
     {"rule": "v1", **score("v1_decision")}]
)
decision_results.to_csv(OUT / "auction_results_baseline_vs_v1.csv", index=False)

print("=" * 70)
print("4. BUY/PASS ON THE 40 AUCTION LOTS (profit = later sale - auction - $1,500)")
print(decision_results.to_string(index=False))
gain = decision_results.loc[1, "total_profit_dollars"] - decision_results.loc[0, "total_profit_dollars"]
print(f"   v1 minus baseline total profit: ${gain:,}")

# ---------------------------------------------------------------- Chart
# Predicted vs actual sale price for the 72 test cars. Dots on the diagonal are
# perfect guesses. The flat gray line is the baseline (same guess for every car).
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"
SERIES_1 = "#2a78d6"

fig, ax = plt.subplots(figsize=(7, 5.6), dpi=200)
lo = min(test_actual.min(), test_v1.min()) - 2_000
hi = max(test_actual.max(), test_v1.max()) + 2_000
ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1, linestyle=(0, (4, 3)), zorder=1,
        label="Perfect guess (predicted = actual)")
ax.axhline(baseline_price, color=MUTED, linewidth=2, zorder=1,
           label=f"Baseline: ${baseline_price:,.0f} for every car")
ax.scatter(test_actual, test_v1, s=36, color=SERIES_1, edgecolor="white", linewidth=1,
           zorder=3, label="v1 prediction (one dot per test car)")

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
money = matplotlib.ticker.FuncFormatter(
    lambda v, _: "$0" if v == 0 else f"{'−' if v < 0 else ''}${abs(v)/1000:.0f}k"
)
ax.xaxis.set_major_formatter(money)
ax.yaxis.set_major_formatter(money)
ax.set_xlabel("Actual sale price", color=INK)
ax.set_ylabel("Predicted sale price", color=INK)
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
ax.tick_params(colors=MUTED)
ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "predicted_vs_actual_test_cars.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- Chart 2
# Total profit on the 40 auction lots, baseline rule vs v1 rule.
rules = ["Baseline rule\n(average price)", "v1 rule\n(regression)"]
profits = decision_results["total_profit_dollars"].tolist()
bought = decision_results["cars_bought"].tolist()
losers = decision_results["money_losing_cars_bought"].tolist()

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
bars = ax.bar(rules, profits, width=0.5, color=[MUTED, SERIES_1], zorder=2)
ax.axhline(0, color=INK, linewidth=1, zorder=3)
span = max(profits) - min(profits)
for bar, p, b, l in zip(bars, profits, bought, losers):
    sign = "+" if p > 0 else "−"
    offset = span * 0.04
    y = p + offset if p >= 0 else p - offset
    ax.text(bar.get_x() + bar.get_width() / 2, y, f"{sign}${abs(p):,}",
            ha="center", va="bottom" if p >= 0 else "top",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(bar.get_x() + bar.get_width() / 2, -span * 0.02 if p >= 0 else span * 0.02,
            f"{b} bought, {l} lost money",
            ha="center", va="top" if p >= 0 else "bottom", fontsize=9, color=MUTED)
ax.set_ylim(min(profits) - span * 0.25, max(profits) + span * 0.2)
ax.yaxis.set_major_formatter(money)
ax.set_ylabel("Total profit on 40 auction cars", color=INK)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "bottom"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color(AXIS)
ax.tick_params(axis="y", colors=MUTED)
ax.tick_params(axis="x", length=0, colors=INK, labelsize=11, pad=8)
fig.tight_layout()
fig.savefig(OUT / "auction_profit_baseline_vs_v1.png", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------- Table images
def table_image(header, rows, path, bold_last=False, col_widths=None):
    """Draw a clean black-on-white table and save it as a PNG for the slides."""
    n_cols = len(header)
    col_widths = col_widths or [1.0 / n_cols] * n_cols
    fig, ax = plt.subplots(figsize=(8, 0.5 * (len(rows) + 1) + 0.1), dpi=200)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, colWidths=col_widths,
                   cellLoc="left", colLoc="left", loc="center", edges="horizontal")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(AXIS)
        cell.set_linewidth(0.8)
        cell.PAD = 0.04
        if r == 0:
            cell.set_text_props(color=MUTED)
        if bold_last and r == len(rows):
            cell.set_text_props(fontweight="bold", color=INK)
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def dollars(v):
    return f"{'−' if v < 0 else '+'}${abs(v):,.0f}"


base_row, v1_row = fit_results.iloc[0], fit_results.iloc[1]
table_image(
    ["72 test cars", "R²", "Typical error"],
    [["Baseline (average price)", f"{base_row.test_R2:.2f}".replace("-", "−"), f"${base_row.test_RMSE_dollars:,.0f}"],
     ["v1 model", f"{v1_row.test_R2:.2f}", f"${v1_row.test_RMSE_dollars:,.0f}"]],
    OUT / "table_test_fit.png", col_widths=[0.5, 0.2, 0.3],
)

b, v = decision_results.iloc[0], decision_results.iloc[1]
table_image(
    ["40 auction cars", "Bought", "Lost money", "Total profit"],
    [["Baseline (average price)", b.cars_bought, b.money_losing_cars_bought, dollars(b.total_profit_dollars)],
     ["v1 model", v.cars_bought, v.money_losing_cars_bought, dollars(v.total_profit_dollars)],
     ["Difference", "", "", dollars(v.total_profit_dollars - b.total_profit_dollars)]],
    OUT / "table_auction_results.png", bold_last=True, col_widths=[0.44, 0.15, 0.19, 0.22],
)

print("=" * 70)
print(f"Saved chart and tables to {OUT}")
