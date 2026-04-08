# Smart Attendance Risk Predictor System

Flask + scikit-learn web app for attendance risk prediction, analytics, and student records.

## Demo login
- Username: admin
- Password: admin123

## Run locally
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python model/train_model.py`
3. Start the app: `python app.py`
4. Open http://127.0.0.1:5000

## Main routes
- / - landing page
- /login - login form
- /dashboard - prediction dashboard
- /analytics - charts page
- /records - student records table
- /predict - prediction API
