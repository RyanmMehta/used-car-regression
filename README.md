# Used car prices: should the dealer buy this car?

Linear regression that predicts a used car's retail sale price from mileage, age, horsepower and accident history, then uses the prediction to decide which auction cars to buy.

## Data

- `data/used_car_sales.csv`: 240 cars the dealer sold in the last year (used to fit and test the model).
- `data/auction_lots.csv`: 40 cars from one recent auction (used only to test the buy/pass rule, never to fit).

## How to reproduce every number

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python analysis.py
```

The 70/30 split uses a fixed random seed (42), so every run gives the same numbers.

## What the script does

1. **Split:** 240 cars into 168 training and 72 test cars.
2. **Fit:** regression of `Sale_Price` on all five features (training cars). Drops any feature with p ≥ 0.05 and refits, then compares adjusted R². `Num_Owners` is dropped (p = 0.87); v1 uses mileage, age, horsepower and accident.
3. **Test fit:** baseline (training average for every car) vs v1 on the 72 test cars: R² and RMSE.
4. **Decision:** `dealer_rule(car, auction_price)` returns `buy` if auction price + $1,500 is below v1's predicted price, otherwise `pass`. `baseline_rule` does the same with the training average. Both run on the 40 auction lots; profit per car bought = `Later_Sale_Price − Auction_Price − 1,500`.

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `analysis_output.txt` | Full printed output, including both regression tables |
| `v1_coefficients.csv` | v1 coefficients, p-values, 95% intervals |
| `test_fit_baseline_vs_v1.csv` | Test-set R² and RMSE, baseline vs v1 |
| `auction_decisions.csv` | Each auction lot: prediction, both decisions, profit if bought |
| `auction_results_baseline_vs_v1.csv` | Cars bought, money-losers, total profit per rule |
| `predicted_vs_actual_test_cars.png` | Chart: predicted vs actual sale price, 72 test cars |
| `auction_profit_baseline_vs_v1.png` | Chart: total profit on the 40 auction lots, baseline rule vs v1 rule |

Both charts are drawn by `analysis.py` (bottom of the file).
