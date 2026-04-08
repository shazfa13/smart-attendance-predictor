from __future__ import annotations

import csv
import io
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from model.train_model import DATASET_PATH, MODEL_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES, train_and_save_model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "attendance_model.pkl"

app = Flask(__name__)
app.secret_key = "smart-attendance-risk-predictor-demo-key"

DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"

SAMPLE_STUDENTS = [
    {"name": "Aarav Sharma", "attendance": 92, "risk": "Low"},
    {"name": "Diya Patel", "attendance": 81, "risk": "Low"},
    {"name": "Ishaan Verma", "attendance": 74, "risk": "Medium"},
    {"name": "Ananya Singh", "attendance": 68, "risk": "High"},
    {"name": "Rahul Mehta", "attendance": 88, "risk": "Low"},
    {"name": "Priya Nair", "attendance": 77, "risk": "Medium"},
    {"name": "Kabir Khan", "attendance": 59, "risk": "High"},
    {"name": "Meera Joshi", "attendance": 95, "risk": "Low"},
    {"name": "Saanvi Gupta", "attendance": 71, "risk": "Medium"},
    {"name": "Arjun Roy", "attendance": 63, "risk": "High"},
]

WEEKLY_ATTENDANCE = [88, 86, 84, 81, 79, 76, 78]
WEEKLY_RISK = [12, 14, 16, 19, 21, 24, 22]


def ensure_model() -> None:
    if not MODEL_FILE.exists():
        train_and_save_model()


ensure_model()
with MODEL_FILE.open("rb") as model_file:
    model = pickle.load(model_file)


@app.context_processor
def inject_globals() -> dict[str, Any]:
    return {"current_year": datetime.now().year}


@app.route("/")
def index() -> str:
    return render_template("index.html", session_user=session.get("user"))


@app.route("/login", methods=["GET", "POST"])
def login() -> Response | str:
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            session["user"] = username
            flash("Login successful. Welcome back.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid credentials. Use admin / admin123.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout() -> Response:
    session.pop("user", None)
    flash("You have been signed out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard() -> Response | str:
    if not session.get("user"):
        flash("Please log in to access the dashboard.", "warning")
        return redirect(url_for("login"))

    records = get_records()
    safe_count = sum(1 for student in records if student["risk"] == "Low")
    warning_count = sum(1 for student in records if student["risk"] == "Medium")
    danger_count = sum(1 for student in records if student["risk"] == "High")

    return render_template(
        "dashboard.html",
        user=session.get("user"),
        student_count=len(records),
        safe_count=safe_count,
        warning_count=warning_count,
        danger_count=danger_count,
        weekly_attendance=WEEKLY_ATTENDANCE,
        weekly_risk=WEEKLY_RISK,
    )


@app.route("/analytics")
def analytics() -> Response | str:
    if not session.get("user"):
        flash("Please log in to access analytics.", "warning")
        return redirect(url_for("login"))

    records = get_records()
    labels = [student["name"] for student in records]
    attendance = [student["attendance"] for student in records]
    risk_values = [student["risk"] for student in records]

    return render_template(
        "analytics.html",
        user=session.get("user"),
        labels=labels,
        attendance=attendance,
        risk_values=risk_values,
        weekly_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        weekly_attendance=WEEKLY_ATTENDANCE,
        weekly_risk=WEEKLY_RISK,
    )


@app.route("/records")
def records() -> Response | str:
    if not session.get("user"):
        flash("Please log in to access records.", "warning")
        return redirect(url_for("login"))

    search = request.args.get("search", "").strip().lower()
    selected_risk = request.args.get("risk", "all")
    records = get_records()

    filtered_records = []
    for student in records:
        if search and search not in student["name"].lower():
            continue
        if selected_risk != "all" and student["risk"] != selected_risk:
            continue
        filtered_records.append(student)

    return render_template(
        "records.html",
        user=session.get("user"),
        records=filtered_records,
        search=search,
        selected_risk=selected_risk,
    )


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    if not session.get("user"):
        return jsonify({"success": False, "message": "Authentication required."}), 401

    payload = request.get_json(silent=True) or request.form
    try:
        total_classes = int(payload.get("total_classes", 0))
        attended_classes = int(payload.get("attended_classes", 0))
        marks = float(payload.get("marks", 0))
        subject_difficulty = str(payload.get("subject_difficulty", "Medium"))
        attendance_trend = str(payload.get("attendance_trend", "Stable"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Please provide valid numeric values."}), 400

    if total_classes <= 0:
        return jsonify({"success": False, "message": "Total classes must be greater than zero."}), 400
    if attended_classes < 0 or attended_classes > total_classes:
        return jsonify({"success": False, "message": "Attended classes must be between 0 and total classes."}), 400

    attendance_percentage = round((attended_classes / total_classes) * 100, 2)

    input_frame = pd.DataFrame(
        [
            {
                "total_classes": total_classes,
                "attended_classes": attended_classes,
                "attendance_percentage": attendance_percentage,
                "marks": marks,
                "subject_difficulty": subject_difficulty,
                "attendance_trend": attendance_trend,
            }
        ],
        columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    )

    probability_risk = float(model.predict_proba(input_frame)[0][1])
    risk_level = classify_risk(probability_risk, attendance_percentage)
    recommendation = build_recommendation(
        total_classes=total_classes,
        attended_classes=attended_classes,
        attendance_percentage=attendance_percentage,
        probability_risk=probability_risk,
        marks=marks,
        subject_difficulty=subject_difficulty,
        attendance_trend=attendance_trend,
    )

    return jsonify(
        {
            "success": True,
            "risk_level": risk_level,
            "probability": round(probability_risk * 100, 2),
            "attendance_percentage": attendance_percentage,
            "recommendation": recommendation,
        }
    )


@app.route("/download-report")
def download_report() -> Response:
    if not session.get("user"):
        flash("Please log in to download reports.", "warning")
        return redirect(url_for("login"))

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["name", "attendance", "risk"])
    writer.writeheader()
    for student in get_records():
        writer.writerow(student)

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=attendance_report.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


@app.route("/api/summary")
def api_summary() -> Response:
    if not session.get("user"):
        return jsonify({"success": False, "message": "Authentication required."}), 401

    records = get_records()
    return jsonify(
        {
            "success": True,
            "total_students": len(records),
            "safe": sum(1 for student in records if student["risk"] == "Low"),
            "warning": sum(1 for student in records if student["risk"] == "Medium"),
            "danger": sum(1 for student in records if student["risk"] == "High"),
        }
    )


def classify_risk(probability_risk: float, attendance_percentage: float) -> str:
    if attendance_percentage < 75 or probability_risk >= 0.7:
        return "High"
    if probability_risk >= 0.35 or attendance_percentage < 82:
        return "Medium"
    return "Low"


def build_recommendation(
    *,
    total_classes: int,
    attended_classes: int,
    attendance_percentage: float,
    probability_risk: float,
    marks: float,
    subject_difficulty: str,
    attendance_trend: str,
) -> str:
    safe_threshold = 75
    classes_needed = max(
        0,
        int(
            ((safe_threshold / 100) * total_classes - attended_classes)
            / (1 - (safe_threshold / 100))
            + 0.9999
        ),
    )

    recommendations = []
    if attendance_percentage < 75:
        recommendations.append(f"You need to attend about {classes_needed} more classes to stay safe.")
    elif attendance_percentage < 85:
        recommendations.append("Maintain consistency to protect your attendance buffer.")
    else:
        recommendations.append("Great attendance. Keep the same pace to stay above the threshold.")

    if probability_risk >= 0.7:
        recommendations.append("Your attendance is critical. Meet your class mentor this week.")
    elif probability_risk >= 0.35:
        recommendations.append("You are in a warning zone. Avoid missing the next few classes.")
    else:
        recommendations.append("Your profile is stable, but consistency is still important.")

    if attendance_trend == "Declining":
        recommendations.append("Your trend is declining, so set a fixed attendance goal for the next two weeks.")
    elif attendance_trend == "Improving":
        recommendations.append("Your trend is improving. Continue this momentum.")

    if subject_difficulty == "Hard" and marks < 60:
        recommendations.append("The subject looks challenging. Consider peer study sessions or tutoring support.")
    elif marks >= 80:
        recommendations.append("Strong marks suggest you can stay balanced with consistent class participation.")

    return " ".join(recommendations)


def get_records() -> list[dict[str, Any]]:
    return SAMPLE_STUDENTS.copy()


if __name__ == "__main__":
    app.run(debug=True)
