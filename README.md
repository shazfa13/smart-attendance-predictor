# Smart Attendance Risk Predictor

A Flask-based ML web application that predicts attendance risk, stores student records in SQLite, and provides live analytics dashboards.

## What This Project Does

- Predicts student risk level (Low, Medium, High) using a trained scikit-learn model.
- Stores student entries in a local SQLite database.
- Visualizes attendance and risk trends in dashboard and analytics charts.
- Supports searching, filtering, exporting, and deleting student records.

## Tech Stack

- Backend: Flask
- Machine Learning: scikit-learn, pandas, numpy
- Database: SQLite (attendance.db)
- Frontend: Jinja2 templates, Bootstrap, Chart.js, vanilla JavaScript

## Core Features

- Login-protected dashboard and analytics pages
- Real-time prediction endpoint
- Separate Add Student page from sidebar navigation
- Automatic attendance percentage calculation
- Database-backed records (no hardcoded sample data)
- Weekly chart simulation from latest 7 database entries
- CSV export for records
- Record deletion from Student Records page

## Project Structure

```
app.py                  # Main Flask app (routes, DB operations, prediction flow)
requirements.txt        # Python dependencies
data/dataset.csv        # Training dataset
model/train_model.py    # Model training script
model/attendance_model.pkl
static/css/style.css
static/js/main.js
templates/*.html
attendance.db           # Auto-created SQLite database
```

## Data Model (SQLite)

Table: students

- id (INTEGER PRIMARY KEY AUTOINCREMENT)
- name (TEXT)
- total_classes (INTEGER)
- attended_classes (INTEGER)
- attendance_percentage (REAL)
- marks (REAL)
- subject_difficulty (TEXT)
- attendance_trend (TEXT)
- risk (TEXT)
- created_at (TIMESTAMP)

## Setup and Run

1. Create and activate a virtual environment.
2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

3. (Optional) Train model manually:

	```bash
	python model/train_model.py
	```

	Note: If model/attendance_model.pkl is missing, the app will auto-train on startup.

4. Start the app:

	```bash
	python app.py
	```

5. Open:

	http://127.0.0.1:5000

## Demo Login

- Username: admin
- Password: admin123

## Main Routes

- / : Landing page
- /login : Login
- /logout : Logout
- /dashboard : Prediction dashboard
- /student-entry : Add student form (stores in DB)
- /analytics : Charts and insights
- /records : Student table with search/filter/delete
- /predict (POST) : Prediction API (no DB insert)
- /add-student (POST) : Add + predict + save to DB
- /download-report : CSV export
- /api/summary : Risk summary stats
- /api/risk-distribution : Risk distribution counts
- /records/delete/<id> and /api/records/<id> : Delete record endpoints

## Notes

- attendance.db is local and should not be committed.
- Configure Flask secret key and credentials for production use.
