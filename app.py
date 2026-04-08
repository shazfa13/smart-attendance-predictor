from __future__ import annotations

import csv
import io
import pickle
import sqlite3
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
    session,
    url_for,
)

from model.train_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, train_and_save_model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "attendance_model.pkl"
DB_PATH = BASE_DIR / "attendance.db"

app = Flask(__name__)
app.secret_key = "smart-attendance-risk-predictor-demo-key"

DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"

ALLOWED_SUBJECT_DIFFICULTY = {"Easy", "Medium", "Hard"}
ALLOWED_ATTENDANCE_TREND = {"Improving", "Stable", "Declining"}


def ensure_model() -> None:
    if not MODEL_FILE.exists():
        train_and_save_model()


ensure_model()
with MODEL_FILE.open("rb") as model_file:
    model = pickle.load(model_file)


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                total_classes INTEGER NOT NULL,
                attended_classes INTEGER NOT NULL,
                attendance_percentage REAL NOT NULL,
                marks REAL NOT NULL,
                subject_difficulty TEXT NOT NULL,
                attendance_trend TEXT NOT NULL,
                risk TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def insert_student(
    *,
    name: str,
    total_classes: int,
    attended_classes: int,
    attendance_percentage: float,
    marks: float,
    subject_difficulty: str,
    attendance_trend: str,
    risk: str,
) -> int:
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO students (
                name,
                total_classes,
                attended_classes,
                attendance_percentage,
                marks,
                subject_difficulty,
                attendance_trend,
                risk
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                total_classes,
                attended_classes,
                attendance_percentage,
                marks,
                subject_difficulty,
                attendance_trend,
                risk,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "name": str(row["name"]),
        "attendance": round(float(row["attendance_percentage"]), 2),
        "risk": str(row["risk"]),
        "total_classes": int(row["total_classes"]),
        "attended_classes": int(row["attended_classes"]),
        "marks": round(float(row["marks"]), 2),
        "subject_difficulty": str(row["subject_difficulty"]),
        "attendance_trend": str(row["attendance_trend"]),
        "created_at": str(row["created_at"]),
    }


def get_all_students() -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                name,
                total_classes,
                attended_classes,
                attendance_percentage,
                marks,
                subject_difficulty,
                attendance_trend,
                risk,
                created_at
            FROM students
            ORDER BY id DESC
            """
        ).fetchall()

    return [row_to_record(row) for row in rows]


def get_recent_students(limit: int = 7) -> list[dict[str, Any]]:
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                name,
                total_classes,
                attended_classes,
                attendance_percentage,
                marks,
                subject_difficulty,
                attendance_trend,
                risk,
                created_at
            FROM students
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records = [row_to_record(row) for row in rows]
    records.reverse()
    return records


def delete_student(student_id: int) -> bool:
    with get_db_connection() as connection:
        cursor = connection.execute("DELETE FROM students WHERE id = ?", (student_id,))
        connection.commit()
        return cursor.rowcount > 0


def build_weekly_series() -> tuple[list[float], list[float]]:
    recent_records = get_recent_students(7)
    attendance_values = [float(student["attendance"]) for student in recent_records]
    risk_map = {"Low": 20.0, "Medium": 60.0, "High": 90.0}
    risk_values = [risk_map.get(str(student["risk"]), 0.0) for student in recent_records]

    while len(attendance_values) < 7:
        attendance_values.insert(0, 0.0)
        risk_values.insert(0, 0.0)

    return attendance_values, risk_values


def parse_student_payload(payload: Any, *, require_name: bool) -> tuple[dict[str, Any], str | None]:
    name = str(payload.get("name", "")).strip()
    if require_name and not name:
        return {}, "Name is required."

    try:
        total_classes = int(payload.get("total_classes", 0))
        attended_classes = int(payload.get("attended_classes", 0))
        marks = float(payload.get("marks", 0))
        subject_difficulty = str(payload.get("subject_difficulty", "Medium")).strip()
        attendance_trend = str(payload.get("attendance_trend", "Stable")).strip()
    except (TypeError, ValueError):
        return {}, "Please provide valid numeric values."

    if total_classes <= 0:
        return {}, "Total classes must be greater than zero."
    if attended_classes < 0 or attended_classes > total_classes:
        return {}, "Attended classes must be between 0 and total classes."
    if marks < 0 or marks > 100:
        return {}, "Marks must be between 0 and 100."
    if subject_difficulty not in ALLOWED_SUBJECT_DIFFICULTY:
        return {}, "Subject difficulty must be Easy, Medium, or Hard."
    if attendance_trend not in ALLOWED_ATTENDANCE_TREND:
        return {}, "Attendance trend must be Improving, Stable, or Declining."

    if not name:
        name = f"Student {datetime.now().strftime('%Y%m%d%H%M%S')}"

    attendance_percentage = round((attended_classes / total_classes) * 100, 2)
    parsed = {
        "name": name,
        "total_classes": total_classes,
        "attended_classes": attended_classes,
        "marks": marks,
        "subject_difficulty": subject_difficulty,
        "attendance_trend": attendance_trend,
        "attendance_percentage": attendance_percentage,
    }
    return parsed, None


def build_prediction(parsed: dict[str, Any]) -> dict[str, Any]:
    input_frame = pd.DataFrame(
        [
            {
                "total_classes": parsed["total_classes"],
                "attended_classes": parsed["attended_classes"],
                "attendance_percentage": parsed["attendance_percentage"],
                "marks": parsed["marks"],
                "subject_difficulty": parsed["subject_difficulty"],
                "attendance_trend": parsed["attendance_trend"],
            }
        ],
        columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    )

    probability_risk = float(model.predict_proba(input_frame)[0][1])
    risk_level = classify_risk(probability_risk, float(parsed["attendance_percentage"]))
    recommendation = build_recommendation(
        total_classes=int(parsed["total_classes"]),
        attended_classes=int(parsed["attended_classes"]),
        attendance_percentage=float(parsed["attendance_percentage"]),
        probability_risk=probability_risk,
        marks=float(parsed["marks"]),
        subject_difficulty=str(parsed["subject_difficulty"]),
        attendance_trend=str(parsed["attendance_trend"]),
    )

    return {
        "risk_level": risk_level,
        "probability": round(probability_risk * 100, 2),
        "attendance_percentage": float(parsed["attendance_percentage"]),
        "recommendation": recommendation,
    }


def predict_and_store(parsed: dict[str, Any]) -> dict[str, Any]:
    prediction = build_prediction(parsed)

    student_id = insert_student(
        name=str(parsed["name"]),
        total_classes=int(parsed["total_classes"]),
        attended_classes=int(parsed["attended_classes"]),
        attendance_percentage=float(parsed["attendance_percentage"]),
        marks=float(parsed["marks"]),
        subject_difficulty=str(parsed["subject_difficulty"]),
        attendance_trend=str(parsed["attendance_trend"]),
        risk=prediction["risk_level"],
    )

    prediction["student_id"] = student_id
    return prediction


init_db()


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
    weekly_attendance, weekly_risk = build_weekly_series()

    return render_template(
        "dashboard.html",
        user=session.get("user"),
        student_count=len(records),
        safe_count=safe_count,
        warning_count=warning_count,
        danger_count=danger_count,
        weekly_attendance=weekly_attendance,
        weekly_risk=weekly_risk,
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
    weekly_attendance, weekly_risk = build_weekly_series()
    risk_distribution = {
        "Low": sum(1 for value in risk_values if value == "Low"),
        "Medium": sum(1 for value in risk_values if value == "Medium"),
        "High": sum(1 for value in risk_values if value == "High"),
    }

    return render_template(
        "analytics.html",
        user=session.get("user"),
        labels=labels,
        attendance=attendance,
        risk_values=risk_values,
        weekly_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        weekly_attendance=weekly_attendance,
        weekly_risk=weekly_risk,
        risk_distribution=risk_distribution,
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


@app.route("/student-entry", methods=["GET", "POST"])
def student_entry() -> Response | str:
    if not session.get("user"):
        flash("Please log in to add student records.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        parsed, error_message = parse_student_payload(request.form, require_name=True)
        if error_message:
            flash(error_message, "danger")
            return render_template("student_entry.html", user=session.get("user"), form_data=request.form)

        try:
            prediction = predict_and_store(parsed)
        except Exception:
            flash("Unable to save student record right now.", "danger")
            return render_template("student_entry.html", user=session.get("user"), form_data=request.form)

        flash(
            f"Student added successfully. Risk: {prediction['risk_level']} ({prediction['probability']}%).",
            "success",
        )
        return redirect(url_for("analytics"))

    return render_template("student_entry.html", user=session.get("user"), form_data={})


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    if not session.get("user"):
        return jsonify({"success": False, "message": "Authentication required."}), 401

    payload = request.get_json(silent=True) or request.form
    parsed, error_message = parse_student_payload(payload, require_name=False)
    if error_message:
        return jsonify({"success": False, "message": error_message}), 400

    try:
        prediction = build_prediction(parsed)
    except Exception:
        return jsonify({"success": False, "message": "Prediction failed. Please try again."}), 500

    return jsonify(
        {
            "success": True,
            "risk_level": prediction["risk_level"],
            "probability": prediction["probability"],
            "attendance_percentage": prediction["attendance_percentage"],
            "recommendation": prediction["recommendation"],
        }
    )


@app.route("/add-student", methods=["POST"])
def add_student() -> Response:
    if not session.get("user"):
        return jsonify({"success": False, "message": "Authentication required."}), 401

    payload = request.get_json(silent=True) or request.form
    parsed, error_message = parse_student_payload(payload, require_name=True)
    if error_message:
        return jsonify({"success": False, "message": error_message}), 400

    try:
        prediction = predict_and_store(parsed)
    except Exception:
        return jsonify({"success": False, "message": "Unable to add student right now."}), 500

    return jsonify(
        {
            "success": True,
            "student_id": prediction["student_id"],
            "name": parsed["name"],
            "risk_level": prediction["risk_level"],
            "attendance_percentage": prediction["attendance_percentage"],
            "probability": prediction["probability"],
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


@app.route("/api/risk-distribution")
def api_risk_distribution() -> Response:
    if not session.get("user"):
        return jsonify({"success": False, "message": "Authentication required."}), 401

    records = get_records()
    return jsonify(
        {
            "success": True,
            "labels": ["Low", "Medium", "High"],
            "values": [
                sum(1 for student in records if student["risk"] == "Low"),
                sum(1 for student in records if student["risk"] == "Medium"),
                sum(1 for student in records if student["risk"] == "High"),
            ],
        }
    )


@app.route("/records/delete/<int:student_id>", methods=["POST", "DELETE"])
@app.route("/api/records/<int:student_id>", methods=["POST", "DELETE"])
def remove_record(student_id: int) -> Response:
    if not session.get("user"):
        return jsonify({"success": False, "message": "Authentication required."}), 401

    try:
        was_deleted = delete_student(student_id)
    except Exception:
        return jsonify({"success": False, "message": "Unable to delete record."}), 500

    if not was_deleted:
        return jsonify({"success": False, "message": "Record not found."}), 404

    return jsonify({"success": True, "message": "Record deleted."})


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
    try:
        return get_all_students()
    except Exception:
        return []


if __name__ == "__main__":
    app.run(debug=True)
