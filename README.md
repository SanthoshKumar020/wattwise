# ⚡ WattWise — AI-Based Electricity Usage Anomaly Detector

An MCA mini project: a Flask web app that learns a household's normal daily
electricity usage, flags abnormal days with machine learning, forecasts the
next 7 days, and estimates the monthly bill.

## Features

- **Accounts** — Register / Login / Logout with SQLite storage and
  werkzeug-hashed passwords. Every user's readings are fully private.
- **Readings** — Add one reading per day (date + units in kWh), update by
  re-adding the same date, delete readings.
- **One-click sample data** — “Load sample data” generates ~60 days of
  realistic household usage (base ≈ 5 units/day, higher on weekends, gentle
  noise, a slow upward drift) with 5 deliberately injected spike days
  (≈ 12–14 units) and runs the analysis immediately.
- **ML anomaly detection** — see below.
- **7-day forecast** — Linear Regression predicts the next 7 days of usage.
- **Bill estimation** — the current month's units are priced through a
  progressive, clearly labelled *indicative* domestic slab table
  (`SLABS` in `app.py`, plus a fixed charge). It is an estimate for
  planning only — not any specific utility's tariff.
- **Alerts** — every anomalous day with units, % above the user's average,
  anomaly score, and a rule-based suggestion of what to check.
- **Dashboard** — Chart.js line chart with anomalous days highlighted in
  red, stat cards (month total, daily average, highest day, anomaly count,
  predicted tomorrow, estimated bill), forecast and bill-breakdown tables,
  energy-saving tips, and a full history table with Normal/Abnormal badges.

## Machine-learning approach

**Anomaly detection — Isolation Forest** (`sklearn.ensemble.IsolationForest`),
retrained on the logged-in user's own readings on *every* analysis run:

- Features per day: `units`, 3-day rolling mean, 7-day rolling mean,
  day-of-week (standardised with `StandardScaler`).
- `contamination=0.1`, 200 trees, fixed `random_state` for reproducibility.
- The stored anomaly score is the negated `decision_function` value, so a
  **higher score = more anomalous**.

**Fallback rule** — with fewer than 14 readings an Isolation Forest is not
meaningful, so WattWise uses a transparent **mean + 2σ rule** (days above
mean + 2 standard deviations are flagged, score = z-score) and the UI
labels the method used, so results are never misrepresented.

**Forecast** — `sklearn.linear_model.LinearRegression` of units against
the day index, extrapolated 7 days past the last reading.

## Sample-data caveat

The sample dataset is **synthetic demo data** generated in code
(`generate_sample_readings` in `app.py`, seeded per user for stability).
It exists so the demo works instantly; real deployments would use actual
meter readings entered by the user.

## Run locally

```bash
python -m venv venv && source venv/bin/activate   # or use any Python 3.10+ env
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

Then: Register → Readings → **Load sample data** → Dashboard / Alerts.

## Deployment

The layout mirrors a standard Flask-on-Vercel setup: `api/index.py`
exposes the Flask `app`, and `vercel.json` bundles `templates/` and
`static/` with the serverless function. On Vercel the SQLite database
lives in `/tmp` (demo-grade, ephemeral persistence); locally it is
`wattwise.db` next to `app.py`.

## Tech stack

Python · Flask · SQLite · scikit-learn (Isolation Forest, Linear
Regression) · NumPy · Pandas · Chart.js
