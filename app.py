"""
WattWise - AI-Based Electricity Usage Anomaly Detector
Flask web app: Home / Dashboard / Readings / Alerts with login.

Users record daily electricity usage (units, kWh). On every analysis run a
scikit-learn Isolation Forest is retrained on the user's own readings
(features: units, 3-day rolling mean, 7-day rolling mean, day-of-week) and
flags abnormal days. With fewer than 14 readings a transparent
mean + 2*std rule is used instead and labelled as such in the UI.
A Linear Regression on the day index predicts the next 7 days, and a
slab-based estimator gives an indicative monthly bill.
"""
import os
import random
import sqlite3
from datetime import date, timedelta
from functools import wraps

import numpy as np
from flask import (Flask, flash, g, redirect, render_template, request,
                   session, url_for)
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from werkzeug.security import check_password_hash, generate_password_hash

BASE = os.path.dirname(os.path.abspath(__file__))
# On Vercel the project directory is read-only, so the SQLite DB lives in /tmp
# (demo-grade persistence). Locally it sits next to app.py.
DB = "/tmp/wattwise.db" if os.environ.get("VERCEL") else os.path.join(BASE, "wattwise.db")

app = Flask(__name__,
            template_folder=os.path.join(BASE, "templates"),
            static_folder=os.path.join(BASE, "static"))
app.secret_key = os.environ.get("WATTWISE_SECRET", "dev-secret-change-me")

MIN_READINGS_FOR_IFOREST = 14

# ---------------------------------------------------------------- bill slabs
# Indicative domestic electricity slab table (per month), defined in ONE
# place. Each entry: (cumulative upper limit in units, rate Rs per unit).
# Units above the last limit are charged at the last rate. This is an
# ESTIMATE for planning only - it is not any specific utility's tariff.
SLABS = [
    (100, 2.25),
    (200, 4.50),
    (400, 6.00),
    (600, 7.50),
    (None, 8.50),   # above 600 units
]
FIXED_CHARGE = 50.0  # indicative fixed / service charge per month (Rs)

STATIC_TIPS = [
    "Set the AC thermostat to 24-26 degC - every degree lower can add roughly 6% to cooling cost.",
    "Switch off appliances at the socket; TVs, set-top boxes and chargers draw standby power all day.",
    "Run washing machines and dishwashers with full loads, preferably outside peak evening hours.",
    "Replace old incandescent/CFL bulbs with LED - lighting is an easy 5-10% of a home bill.",
    "Service the fridge door gasket and keep coils dust-free; a struggling fridge runs its compressor longer.",
    "Use natural light and ventilation on mild days instead of lights + fans + AC together.",
]


# ---------------------------------------------------------------- database
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def _close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    conn = sqlite3.connect(DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            date TEXT NOT NULL,
            units REAL NOT NULL,
            is_anomaly INTEGER,
            anomaly_score REAL,
            method TEXT,
            UNIQUE(user_id, date)
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- auth
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warn")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 6:
            flash("Fill all fields; password must be at least 6 characters.", "error")
            return render_template("register.html")
        try:
            db().execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, generate_password_hash(password)))
            db().commit()
        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "error")
            return render_template("register.html")
        flash("Account created - please log in.", "ok")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ---------------------------------------------------------------- helpers
def get_readings(user_id):
    return db().execute(
        "SELECT * FROM readings WHERE user_id = ? ORDER BY date ASC",
        (user_id,)).fetchall()


def build_features(rows):
    """Feature matrix: [units, 3-day rolling mean, 7-day rolling mean, dow]."""
    units = np.array([r["units"] for r in rows], dtype=float)
    n = len(units)
    roll3 = np.array([units[max(0, i - 2):i + 1].mean() for i in range(n)])
    roll7 = np.array([units[max(0, i - 6):i + 1].mean() for i in range(n)])
    dow = np.array([date.fromisoformat(r["date"]).weekday() for r in rows],
                   dtype=float)
    return np.column_stack([units, roll3, roll7, dow])


def run_analysis(user_id):
    """Retrain on the user's data and store per-reading anomaly results.

    Returns the method label used, or None when there is no data.
    """
    rows = get_readings(user_id)
    if not rows:
        return None
    units = np.array([r["units"] for r in rows], dtype=float)
    if len(rows) >= MIN_READINGS_FOR_IFOREST:
        X = StandardScaler().fit_transform(build_features(rows))
        model = IsolationForest(n_estimators=200, contamination=0.1,
                                random_state=42)
        preds = model.fit_predict(X)
        # decision_function: lower = more anomalous -> negate so that a
        # HIGHER stored score means MORE anomalous.
        scores = -model.decision_function(X)
        flags = (preds == -1)
        method = "Isolation Forest"
    else:
        mean, std = float(units.mean()), float(units.std())
        threshold = mean + 2 * std if std > 0 else float("inf")
        flags = units > threshold
        scores = (units - mean) / std if std > 0 else np.zeros_like(units)
        method = "Mean + 2-sigma rule (not enough data for Isolation Forest)"
    conn = db()
    for row, flag, score in zip(rows, flags, scores):
        conn.execute(
            "UPDATE readings SET is_anomaly = ?, anomaly_score = ?, method = ? "
            "WHERE id = ?",
            (int(bool(flag)), round(float(score), 4), method, row["id"]))
    conn.commit()
    return method


def predict_next_7(rows):
    """Linear Regression on day index -> next 7 days of usage."""
    if len(rows) < 2:
        return []
    units = np.array([r["units"] for r in rows], dtype=float)
    X = np.arange(len(rows), dtype=float).reshape(-1, 1)
    model = LinearRegression().fit(X, units)
    last = date.fromisoformat(rows[-1]["date"])
    out = []
    for k in range(1, 8):
        pred = float(model.predict([[len(rows) + k - 1]])[0])
        out.append({"date": (last + timedelta(days=k)).isoformat(),
                    "units": round(max(pred, 0.1), 2)})
    return out


def estimate_bill(total_units):
    """Progressive slab estimate. Returns (total_rs, breakdown list)."""
    cost = FIXED_CHARGE
    breakdown = [{"label": "Fixed / service charge", "units": None,
                  "rate": None, "amount": round(FIXED_CHARGE, 2)}]
    prev = 0.0
    remaining = float(total_units)
    for limit, rate in SLABS:
        if remaining <= 0:
            break
        width = remaining if limit is None else min(remaining, limit - prev)
        if width > 0:
            label = (f"Units {int(prev) + 1}-{int(limit)}" if limit
                     else f"Units above {int(prev)}")
            breakdown.append({"label": label, "units": round(width, 2),
                              "rate": rate, "amount": round(width * rate, 2)})
            cost += width * rate
            remaining -= width
        if limit is not None:
            prev = float(limit)
    return round(cost, 2), breakdown


def suggestion_for(units, avg):
    pct = (units - avg) / avg * 100 if avg > 0 else 0.0
    if pct >= 100:
        text = ("Large spike - check high-load appliances: AC, water heater / "
                "geyser, fridge compressor, or an appliance fault.")
    elif pct >= 40:
        text = ("Moderate spike - check for appliances left running: AC left "
                "on, iron, oven, washing machine back-to-back cycles.")
    else:
        text = ("Slightly above your pattern - worth a quick look for lights, "
                "fans or standby loads left on.")
    return pct, text


def generate_sample_readings(user_id):
    """~60 days of realistic household usage with 5 injected spike anomalies.

    Base ~5 units/day, slightly higher on weekends, gentle noise and a slow
    upward drift. Deterministic per user (seeded) so the demo is stable.
    """
    rng = random.Random(1000 + user_id)
    today = date.today()
    start = today - timedelta(days=59)
    spikes = {7: 12.6, 19: 13.4, 31: 12.1, 44: 14.0, 53: 12.9}
    rows = []
    for i in range(60):
        d = start + timedelta(days=i)
        base = 5.0 + 0.01 * i
        if d.weekday() >= 5:
            base += 1.1
        if d.weekday() == 2:
            base -= 0.5
        units = base + rng.uniform(-1.0, 1.0)
        if i in spikes:
            units = spikes[i] + rng.uniform(-0.3, 0.3)
        rows.append((d.isoformat(), round(max(units, 1.5), 2)))
    return rows


# ---------------------------------------------------------------- pages
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
@login_required
def dashboard():
    uid = session["user_id"]
    rows = get_readings(uid)
    method = rows[0]["method"] if rows and rows[0]["method"] else None
    predictions = predict_next_7(rows) if len(rows) >= 2 else []
    stats = None
    bill = None
    alerts_preview = []
    if rows:
        units = [r["units"] for r in rows]
        avg = sum(units) / len(units)
        month_prefix = date.today().strftime("%Y-%m")
        month_units = sum(r["units"] for r in rows
                          if r["date"].startswith(month_prefix))
        highest = max(rows, key=lambda r: r["units"])
        anomalies = [r for r in rows if r["is_anomaly"]]
        stats = {
            "month_units": round(month_units, 2),
            "daily_avg": round(avg, 2),
            "highest_date": highest["date"],
            "highest_units": highest["units"],
            "anomaly_count": len(anomalies),
            "predicted_tomorrow": predictions[0]["units"] if predictions else None,
        }
        bill_total, breakdown = estimate_bill(month_units)
        bill = {"total": bill_total, "breakdown": breakdown,
                "month_units": round(month_units, 2)}
        for r in anomalies:
            pct, text = suggestion_for(r["units"], avg)
            alerts_preview.append({"date": r["date"], "units": r["units"],
                                   "pct": round(pct, 1), "text": text})
    chart = {
        "labels": [r["date"] for r in rows],
        "units": [r["units"] for r in rows],
        "anomaly": [bool(r["is_anomaly"]) for r in rows],
    }
    return render_template("dashboard.html", rows=list(reversed(rows)),
                           stats=stats, bill=bill, method=method,
                           predictions=predictions, chart=chart,
                           tips=STATIC_TIPS, alerts=alerts_preview)


@app.route("/readings")
@login_required
def readings():
    rows = get_readings(session["user_id"])
    return render_template("readings.html", rows=list(reversed(rows)),
                           today=date.today().isoformat())


@app.route("/readings/add", methods=["POST"])
@login_required
def add_reading():
    uid = session["user_id"]
    d = request.form.get("date", "").strip()
    try:
        date.fromisoformat(d)
        units = float(request.form.get("units", ""))
        if units < 0:
            raise ValueError
    except (TypeError, ValueError):
        flash("Enter a valid date and a non-negative units value.", "error")
        return redirect(url_for("readings"))
    db().execute(
        "INSERT INTO readings (user_id, date, units) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id, date) DO UPDATE SET units = excluded.units, "
        "is_anomaly = NULL, anomaly_score = NULL, method = NULL",
        (uid, d, units))
    db().commit()
    flash(f"Reading for {d} saved ({units} units).", "ok")
    return redirect(url_for("readings"))


@app.route("/readings/delete/<int:reading_id>", methods=["POST"])
@login_required
def delete_reading(reading_id):
    db().execute("DELETE FROM readings WHERE id = ? AND user_id = ?",
                 (reading_id, session["user_id"]))
    db().commit()
    flash("Reading deleted.", "ok")
    return redirect(url_for("readings"))


@app.route("/readings/load_sample", methods=["POST"])
@login_required
def load_sample():
    uid = session["user_id"]
    conn = db()
    conn.execute("DELETE FROM readings WHERE user_id = ?", (uid,))
    conn.executemany(
        "INSERT INTO readings (user_id, date, units) VALUES (?, ?, ?)",
        [(uid, d, u) for d, u in generate_sample_readings(uid)])
    conn.commit()
    run_analysis(uid)
    flash("Sample data loaded: 60 days of household usage (synthetic demo "
          "data with a few built-in spikes). Analysis has been run.", "ok")
    return redirect(url_for("dashboard"))


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    method = run_analysis(session["user_id"])
    if method is None:
        flash("Add some readings first, then run the analysis.", "warn")
        return redirect(url_for("readings"))
    flash(f"Analysis complete - method used: {method}.", "ok")
    return redirect(url_for("dashboard"))


@app.route("/alerts")
@login_required
def alerts():
    uid = session["user_id"]
    rows = get_readings(uid)
    avg = (sum(r["units"] for r in rows) / len(rows)) if rows else 0.0
    items = []
    for r in rows:
        if r["is_anomaly"]:
            pct, text = suggestion_for(r["units"], avg)
            items.append({"date": r["date"], "units": r["units"],
                          "score": r["anomaly_score"],
                          "pct": round(pct, 1), "text": text})
    method = rows[0]["method"] if rows and rows[0]["method"] else None
    return render_template("alerts.html", alerts=list(reversed(items)),
                           avg=round(avg, 2), method=method)


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
