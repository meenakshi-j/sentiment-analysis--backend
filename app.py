from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
from config import get_db_connection
from flask_mail import Mail, Message
import random
from datetime import datetime, timedelta
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import mysql.connector
from textblob import TextBlob
from google_play_scraper import reviews, Sort
import schedule
import time
import pandas as pd
from aspect_config import ASPECT_KEYWORDS
import json
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
import os
import mysql.connector
import pymysql

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")

load_dotenv(env_path)

app = Flask(__name__)

# Allowed apps for comparison by platform
ALLOWED_APPS = {
    "online_shopping": ["amazon", "flipkart", "myntra", "ajio", "meesho"],
    "instant_store": ["blinkit", "zepto", "instamart", "bigbasket", "jiomart"],
    "food_delivery": ["swiggy", "zomato", "dominos", "kfc", "mcdonalds"]
}
# -----------------------------
# SENTIMENT FUNCTION
# -----------------------------
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"
    
def detect_aspect(review_text):
    review_text = review_text.lower()

    for aspect, keywords in ASPECT_KEYWORDS.items():
        for word in keywords:
            if word in review_text:
                return aspect

    return "Other"

# -----------------------------
# CHURN RISK CALCULATION
# -----------------------------
def calculate_churn_risk(total_reviews, positive, neutral, negative):

    if total_reviews == 0:
        return 0,0,0,0

    negative_ratio = negative / total_reviews

    # Estimate customer base
    total_customers = total_reviews * 12

    high_risk = int(total_customers * negative_ratio * 0.7)
    medium_risk = int(total_customers * negative_ratio * 0.3)
    low_risk = total_customers - high_risk - medium_risk

    return total_customers, high_risk, medium_risk, low_risk

def calculate_risk_factors(aspect_summary):

    factors = {
        "negative_reviews":0,
        "low_engagement":0,
        "support_tickets":0,
        "payment_issues":0,
        "app_performance":0,
        "response_time":0
    }

    total_negative = 0

    for aspect,data in aspect_summary.items():
        total_negative += data["negative"]

    if total_negative == 0:
        return factors

    for aspect,data in aspect_summary.items():

        neg = data["negative"]
        percentage = int((neg/total_negative)*100)

        if aspect == "Customer Service":
            factors["support_tickets"] = percentage

        elif aspect == "App Performance":
            factors["app_performance"] = percentage

        elif aspect == "Pricing":
            factors["payment_issues"] = percentage

        elif aspect == "Delivery":
            factors["response_time"] = percentage

        elif aspect == "Other":
            factors["low_engagement"] = percentage

    factors["negative_reviews"] = int((total_negative/(total_negative+1))*100)

    return factors
# -----------------------------
# FETCH & STORE PLAY STORE REVIEWS
# -----------------------------
def fetch_and_store_reviews(package_name, app_id):

    result, _ = reviews(
        package_name,
        lang='en',
        country='in',
        sort=Sort.NEWEST,
        count=200
    )

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    positive = 0
    neutral = 0
    negative = 0
    total_rating = 0

    aspect_summary = {}
    
    for r in result:
        review_text = r['content']
        sentiment = analyze_sentiment(review_text)
        aspect = detect_aspect(review_text)
        
        total_rating += r['score']

        # Overall sentiment counters
        if sentiment == "Positive":
            positive += 1
        elif sentiment == "Neutral":
            neutral += 1
        else:
            negative += 1

        # -------------------------
        # Aspect Sentiment Tracking
        # -------------------------

        if aspect not in aspect_summary:
            aspect_summary[aspect] = {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            }
        aspect_summary[aspect][sentiment.lower()] += 1

    total_reviews = len(result)

    avg_rating = round(total_rating / total_reviews, 2) if total_reviews > 0 else 0
    satisfaction = round((positive / total_reviews) * 100, 2) if total_reviews > 0 else 0
    risk_factors = calculate_risk_factors(aspect_summary)
    
    cursor.execute("""
                   INSERT INTO app_metrics
                   (app_id, total_reviews, avg_rating, satisfaction_percentage,
                   positive_count, neutral_count, negative_count,
                   aspect_sentiment, risk_factors)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON DUPLICATE KEY UPDATE
                   total_reviews=%s,
                   avg_rating=%s,
                   satisfaction_percentage=%s,
                   positive_count=%s,
                   neutral_count=%s,
                   negative_count=%s,
                   aspect_sentiment=%s
    """, (
        app_id,
        total_reviews,
        avg_rating,
        satisfaction,
        positive,
        neutral,
        negative,
        json.dumps(aspect_summary),
        json.dumps(risk_factors),

        total_reviews,
        avg_rating,
        satisfaction,
        positive,
        neutral,
        negative,
        json.dumps(aspect_summary),
        json.dumps(risk_factors)
    ))

    connection.commit()
    cursor.close()
    connection.close()
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")

jwt = JWTManager(app)
# Gmail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
app.config['MAIL_DEFAULT_SENDER'] = os.getenv("MAIL_DEFAULT_SENDER")

mail = Mail(app)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
# -----------------------------
# LOAD TRAINED DATASET REVIEWS (LOAD ONCE)
# -----------------------------

# -----------------------------
# REGISTER API
# -----------------------------
@app.route('/register', methods=['POST'])
def register():

    data = request.get_json()

    full_name = data.get("full_name")
    email = data.get("email")
    mobile = data.get("mobile")
    username = data.get("username")
    password = data.get("password")

    if not full_name or not email or not username or not password:
        return jsonify({"message": "Required fields missing"}), 400

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    # Check existing user
    check_query = "SELECT * FROM users WHERE username=%s OR email=%s"
    cursor.execute(check_query, (username, email))
    existing_user = cursor.fetchone()

    if existing_user:
        cursor.close()
        connection.close()
        return jsonify({
            "status": "fail",
            "message": "Username or Email already exists"
        }), 409

    # Insert new user
    insert_query = """
        INSERT INTO users (full_name, email, mobile, username, password)
        VALUES (%s, %s, %s, %s, %s)
    """

    hashed_password = generate_password_hash(password)

    cursor.execute(insert_query, (
        full_name,
        email,
        mobile,
        username,
        hashed_password
    ))

    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({
        "status": "success",
        "message": "Account created successfully"
    })

# -----------------------------
# LOGIN API
# -----------------------------
@app.route('/login', methods=['POST'])
def login():

    data = request.get_json()

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"message": "Missing username or password"}), 400

    connection = get_db_connection()
    cursor = connection.cursor()

    query = "SELECT * FROM users WHERE username=%s OR email=%s"
    cursor.execute(query, (username, username))
    user = cursor.fetchone()

    cursor.close()
    connection.close()

    if user and check_password_hash(user['password'], password):
        access_token = create_access_token(identity=username)
        return jsonify({
            "status": "success",
            "message": "Login successful",
            "access_token": access_token
        })
    else:
        return jsonify({
            "status": "fail",
            "message": "Invalid username or password"
        }), 401

# -----------------------------
# PREDICT API
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    text = data.get("text")
    username = data.get("username")

    if not text or not username:
        return jsonify({"error": "Missing text or username"}), 400

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    insert_query = """
        INSERT INTO sentiment_history (username, text, sentiment)
        VALUES (%s, %s, %s)
    """

    cursor.execute(insert_query, (username, text, prediction))
    connection.commit()

    cursor.close()
    connection.close()

    return jsonify({
        "text": text,
        "sentiment": prediction
    })

@app.route('/send-reset-code', methods=['POST'])
def send_reset_code():

    data = request.get_json()
    email = data.get("email")

    if not email:
        return jsonify({"message": "Email is required"}), 400

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    # ✅ Check if user exists
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()

    if not user:
        cursor.close()
        connection.close()
        return jsonify({
            "status": "fail",
            "message": "Email not registered"
        }), 404

    # ---------------------------------------------------
    # ✅ RESEND TIMER CHECK (60 seconds)
    # reset_sent_at column index = 10
    # ---------------------------------------------------

    if user["reset_sent_at"] is not None:
        last_sent = user["reset_sent_at"]
        now = datetime.now()

        time_difference = now - last_sent

        if time_difference < timedelta(seconds=60):
            remaining = 60 - int(time_difference.total_seconds())

            cursor.close()
            connection.close()

            return jsonify({
                "status": "fail",
                "message": f"Please wait {remaining} seconds before requesting again"
            }), 429

    # ---------------------------------------------------
    # ✅ Generate new reset code
    # ---------------------------------------------------

    reset_code = str(random.randint(100000, 999999))
    expiry_time = datetime.now() + timedelta(minutes=5)
    current_time = datetime.now()

    # ---------------------------------------------------
    # ✅ Update database
    # ---------------------------------------------------

    cursor.execute("""
        UPDATE users 
        SET reset_code=%s,
            reset_code_expiry=%s,
            reset_sent_at=%s
        WHERE email=%s
    """, (reset_code, expiry_time, current_time, email))

    connection.commit()
    msg = Message(
        subject="Password Reset Code",
        recipients=[email]
    )
    msg.body = f"Your password reset code is: {reset_code}"
    try:
        mail.send(msg)
    except Exception as e:
        return jsonify({
            "status": "fail",
            "message": str(e)
        }), 500
    cursor.close()
    connection.close()
    
    return jsonify({
        "status": "success",
        "message": "Reset code sent successfully"
    }), 200

@app.route("/verify-reset-code", methods=["POST"])
def verify_reset_code():
    data = request.get_json()
    email = data.get("email")
    reset_code = data.get("reset_code")

    if not email or not reset_code:
        return jsonify({
            "status": "fail",
            "message": "Email and reset code are required"
        }), 400

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    # Check if user exists
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()

    if not user:
        cursor.close()
        connection.close()
        return jsonify({
            "status": "fail",
            "message": "User not found"
        }), 404

    # Check if reset code matches
    if user["reset_code"] != reset_code:
        cursor.close()
        connection.close()
        return jsonify({
            "status": "fail",
            "message": "Invalid reset code"
        }), 400

    cursor.close()
    connection.close()

    return jsonify({
        "status": "success",
        "message": "Code verified successfully"
    }), 200

@app.route('/reset-password', methods=['POST'])
def reset_password():

    data = request.get_json()
    email = data.get("email")
    new_password = data.get("new_password")

    if not email or not new_password:
        return jsonify({"message": "Email and new password required"}), 400

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    

    # Hash new password
    hashed_password = generate_password_hash(new_password)

    # Update password and clear reset code
    cursor.execute("""
        UPDATE users 
        SET password=%s, reset_code=NULL, reset_code_expiry=NULL
        WHERE email=%s
    """, (hashed_password, email))

    connection.commit()

    cursor.close()
    connection.close()

    return jsonify({"message": "Password reset successful"})


@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify({
        "logged_in_as": current_user
    }), 200

# -----------------------------
# GET PROFILE DATA
# -----------------------------
@app.route("/profile", methods=["GET"])
@jwt_required()
def get_profile():

    current_user = get_jwt_identity()

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    query = """
        SELECT username, email, mobile
        FROM users
        WHERE username=%s
    """

    cursor.execute(query, (current_user,))
    user = cursor.fetchone()

    cursor.close()
    connection.close()

    if not user:
        return jsonify({
            "status": "fail",
            "message": "User not found"
        }), 404

    return jsonify({
        "status": "success",
        "username": user["username"],
        "email": user["email"],
        "mobile": user["mobile"]
    })

# -----------------------------
# UPDATE USERNAME
# -----------------------------
@app.route("/update-username", methods=["PUT"])
@jwt_required()
def update_username():

    current_user = get_jwt_identity()

    data = request.get_json()
    new_username = data.get("username")

    if not new_username:
        return jsonify({
            "status": "fail",
            "message": "Username required"
        }), 400

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    # Check if username exists
    cursor.execute("SELECT * FROM users WHERE username=%s", (new_username,))
    existing = cursor.fetchone()

    if existing:
        cursor.close()
        connection.close()

        return jsonify({
            "status": "fail",
            "message": "Username already taken"
        }), 409

    # Update username
    cursor.execute("""
        UPDATE users
        SET username=%s
        WHERE username=%s
    """, (new_username, current_user))

    connection.commit()

    cursor.close()
    connection.close()

    return jsonify({
        "status": "success",
        "message": "Username updated successfully"
    })

# -----------------------------
# CHANGE PASSWORD
# -----------------------------
@app.route("/change-password", methods=["PUT"])
@jwt_required()
def change_password():

    current_user = get_jwt_identity()

    data = request.get_json()

    current_password = data.get("current_password")
    new_password = data.get("new_password")
    confirm_password = data.get("confirm_password")

    if not current_password or not new_password or not confirm_password:
        return jsonify({
            "status": "fail",
            "message": "All fields required"
        }), 400

    if new_password != confirm_password:
        return jsonify({
            "status": "fail",
            "message": "Passwords do not match"
        }), 400

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    # Get user password
    cursor.execute("""
        SELECT password
        FROM users
        WHERE username=%s
    """, (current_user,))

    user = cursor.fetchone()

    if not user:
        cursor.close()
        connection.close()

        return jsonify({
            "status": "fail",
            "message": "User not found"
        }), 404

    # Verify current password
    if not check_password_hash(user['password'], current_password):
        cursor.close()
        connection.close()

        return jsonify({
            "status": "fail",
            "message": "Current password incorrect"
        }), 401

    # Update password
    hashed_password = generate_password_hash(new_password)

    cursor.execute("""
        UPDATE users
        SET password=%s
        WHERE username=%s
    """, (hashed_password, current_user))

    connection.commit()

    cursor.close()
    connection.close()

    return jsonify({
        "status": "success",
        "message": "Password changed successfully"
    })

# -----------------------------
# GET APPS BY PLATFORM
# -----------------------------
@app.route('/platform/<int:platform_id>/apps', methods=['GET'])
def get_apps_by_platform(platform_id):
    try:
        # Map platform_id to platform_type
        platform_map = {
            1: "online_shopping",
            2: "instant_store",
            3: "food_delivery"
        }

        platform_type = platform_map.get(platform_id)
        if not platform_type:
            return jsonify({"status": "fail", "message": "Invalid platform ID"}), 400

        allowed_apps = ALLOWED_APPS[platform_type]

        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # Fetch apps that exist in allowed_apps
        format_strings = ",".join(["%s"] * len(allowed_apps))
        query = f"""
            SELECT app_id, app_name 
            FROM apps 
            WHERE LOWER(app_name) IN ({format_strings})
        """
        cursor.execute(query, tuple([a.lower() for a in allowed_apps]))
        apps = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "platform_id": platform_id,
            "platform_type": platform_type,
            "apps": apps
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# -----------------------------
# GET PLATFORMS API
# -----------------------------
@app.route('/app-analysis/<int:app_id>', methods=['GET'])
def app_analysis(app_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 1️⃣ Get app details
        cursor.execute("SELECT app_name, package_name FROM apps WHERE app_id = %s", (app_id,))
        app_data = cursor.fetchone()

        if not app_data:
            return jsonify({"status": "error", "message": "App not found"})

        app_name = app_data["app_name"]
        package_name = app_data["package_name"]

        

        if not package_name:
            return jsonify({"status": "error", "message": "Package name missing"})

        # 2️⃣ Smart Auto-Map Dataset (Combine filename + multi-app files)

        import glob
        import os
        import re
        
        dataset_reviews = []
        dataset_files_used = []
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_dir, "datasets", "*.csv")
        dataset_files = glob.glob(dataset_path)
        
        normalized_app_name = re.sub(r'[^a-z0-9]', '', app_name.lower())
        for file in dataset_files:
            try:
                df = pd.read_csv(file)
                df.columns = df.columns.str.lower()
                
                filename = os.path.basename(file).lower()
                filename_normalized = re.sub(r'[^a-z0-9]', '', filename)
                
                # Find review column
                possible_review_columns = [
                    'review_text', 'review text',
                    'review', 'reviews',
                    'text', 'comment',
                    'content'   # ✅ ADD THIS LINE
                ]
                
                review_column = None
                for col in possible_review_columns:
                    if col in df.columns:
                        review_column = col
                        break
                    
                if not review_column:
                    continue

                # -----------------------------------------
                # MATCH CONDITION 1: Filename match
                # -----------------------------------------
                if normalized_app_name in filename_normalized:
                    reviews_list = df[review_column].dropna().tolist()
                    dataset_reviews.extend(reviews_list)
                    dataset_files_used.append(filename)
                    print("Matched via filename:", filename)
                    print("Rows found:", len(reviews_list))
                    continue

                # -----------------------------------------
                # MATCH CONDITION 2: App column match
                # -----------------------------------------
                possible_app_columns = [
                    'app_name', 'platform',
                    'agent name', 'agent_name',
                    'application', 'app'
                ]
                
                for col in possible_app_columns:
                    if col in df.columns:
                        series_normalized = (
                            df[col]
                            .astype(str)
                            .str.lower()
                            .apply(lambda x: re.sub(r'[^a-z0-9]', '', x))
                        )
                        
                        matched_rows = df[series_normalized.str.contains(normalized_app_name, na=False)]
                        
                        if not matched_rows.empty:
                            reviews_list = matched_rows[review_column].dropna().tolist()
                            dataset_reviews.extend(reviews_list)
                            dataset_files_used.append(filename)
                            print("Matched via app column:", filename)
                            break

                # -----------------------------------------
                # MATCH CONDITION 3: Review text match
                # -----------------------------------------
                else:
                    matched_reviews = []
                    
                    for review in df[review_column].dropna():
                        review_normalized = re.sub(r'[^a-z0-9]', '', str(review).lower())
                        if normalized_app_name in review_normalized:
                            matched_reviews.append(review)
                            
                    if matched_reviews:
                        dataset_reviews.extend(matched_reviews)
                        dataset_files_used.append(filename)
                        print("Matched via review content:", filename)
                        
            except Exception as e:
                print("Error reading file:", file, str(e))
                continue
            
        # Remove duplicates
        dataset_reviews = list(set(dataset_reviews))

        print("FINAL DATASET COUNT:", len(dataset_reviews))


        if not dataset_reviews:
            print(f"No matching dataset found for {app_name}")

        # Remove duplicate reviews
        dataset_reviews = list(set(dataset_reviews))

        # 3️⃣ Fetch Play Store reviews
        try:
            from google_play_scraper import reviews, Sort
            
            all_results = []
            continuation_token = None
            
            while True:
                result, continuation_token = reviews(
                    package_name,
                    lang='en',
                    country='in',
                    sort=Sort.NEWEST,
                    count=200,
                    continuation_token=continuation_token
                )
                
                if not result:
                    break
                
                all_results.extend(result)
                
                print("Fetched so far:", len(all_results))
                
                if not continuation_token:
                    break

                # Optional limit (remove if you want unlimited)
                if len(all_results) >= 2000:
                    break
                
            result = all_results
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Play Store fetch failed: {str(e)}"
            })

        playstore_reviews = [r['content'] for r in result if r.get('content')]
        # 🔎 DEBUG COUNTS (ADD HERE)
        print("Dataset reviews count:", len(dataset_reviews))
        print("Playstore reviews count:", len(playstore_reviews))

        # 4️⃣ Combine Reviews
        all_reviews = playstore_reviews + dataset_reviews

        positive = 0
        neutral = 0
        negative = 0
        total_rating = 0
        
        # ⭐ NEW: Aspect dictionary to count aspect sentiment
        aspect_summary = {}
        
        # Calculate rating only from Play Store
        for r in result:
            total_rating += r.get('score', 0)

        # Sentiment + Aspect prediction
        for review_text in all_reviews:
            if not review_text or not review_text.strip():
                continue

            # Sentiment prediction
            vector = vectorizer.transform([review_text])
            prediction = model.predict(vector)[0]
            sentiment = prediction.lower()

            # Overall count
            if sentiment == "positive":
                positive += 1
            elif sentiment == "neutral":
                neutral += 1
            else:
                negative += 1

            # ⭐ Detect aspect
            aspect = detect_aspect(review_text)

            # Initialize aspect if not exists
            if aspect not in aspect_summary:
                aspect_summary[aspect] = {
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0
                }

            # Count aspect sentiment
            aspect_summary[aspect][sentiment] += 1

        total_reviews = len(all_reviews)

        average_rating = round(total_rating / len(result), 2) if len(result) > 0 else 0
        satisfaction = round((positive / total_reviews) * 100, 2) if total_reviews > 0 else 0
        risk_factors = calculate_risk_factors(aspect_summary)
        # -----------------------------
        # CHURN CALCULATION
        # -----------------------------
        total_customers, high_risk, medium_risk, low_risk = calculate_churn_risk(
            total_reviews,
            positive,
            neutral,
            negative
        )
        
        risk_factors = calculate_risk_factors(aspect_summary)

        # 5️⃣ Store metrics (Update if exists)
        aspect_json = json.dumps(aspect_summary)

        cursor.execute("""
            INSERT INTO app_metrics
            (app_id, total_reviews, average_rating, satisfaction_percentage,
             positive_count, neutral_count, negative_count)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
            total_reviews=%s,
            average_rating=%s,
            satisfaction_percentage=%s,
            positive_count=%s,
            neutral_count=%s,
            negative_count=%s
        """, (
            app_id, total_reviews, average_rating, satisfaction,
            positive, neutral, negative,
            total_reviews, average_rating, satisfaction,
            positive, neutral, negative
        ))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "app_id": app_id,
            "app_name": app_name,
            
            "total_reviews": total_reviews,
            "average_rating": average_rating,
            "satisfaction": satisfaction,
            
            "sentiment_breakdown": {
                "positive": positive,
                "neutral": neutral,
                "negative": negative
            },
            
            "aspect_sentiment": aspect_summary,
            
            "churn_analysis": {
                "total_customers": total_customers,
                "high_risk": high_risk,
                "medium_risk": medium_risk,
                "low_risk": low_risk
            },
            
            "risk_factors": risk_factors
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/compare-apps', methods=['POST'])
def compare_apps():
    data = request.get_json()
    app_ids = data.get("app_ids")  # List of app IDs
    platform_type = data.get("platform_type")  # New field from frontend

    # -----------------------------
    # Validate minimum and maximum apps
    # -----------------------------
    if not app_ids or not isinstance(app_ids, list):
        return jsonify({"status": "fail", "message": "App IDs required"}), 400
    if len(app_ids) < 2:
        return jsonify({"status": "fail", "message": "Select minimum 2 apps for comparison"}), 400
    if len(app_ids) > 5:
        return jsonify({"status": "fail", "message": "You can select maximum 5 apps for comparison"}), 400

    # -----------------------------
    # Validate apps belong to platform
    # -----------------------------
    if platform_type not in ALLOWED_APPS:
        return jsonify({"status": "fail", "message": "Invalid platform type"}), 400

    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # Get all apps for this platform
    format_strings = ",".join(["%s"] * len(ALLOWED_APPS[platform_type]))
    cursor.execute(f"""
        SELECT app_id, LOWER(app_name) as app_name
        FROM apps
        WHERE LOWER(app_name) IN ({format_strings})
    """, tuple([a.lower() for a in ALLOWED_APPS[platform_type]]))
    platform_apps = cursor.fetchall()
    platform_app_ids = [a['app_id'] for a in platform_apps]

    # Check if selected apps belong to platform
    invalid_apps = [a for a in app_ids if a not in platform_app_ids]
    if invalid_apps:
        cursor.close()
        conn.close()
        return jsonify({
            "status": "fail",
            "message": f"Selected apps do not belong to {platform_type}",
            "invalid_apps": invalid_apps
        }), 400

    # -----------------------------
    # Fetch comparison metrics (existing logic)
    # -----------------------------
    comparison_data = []

    for app_id in app_ids:
        cursor.execute("SELECT app_name, package_name FROM apps WHERE app_id = %s", (app_id,))
        app_info = cursor.fetchone()
        if not app_info:
            continue

        cursor.execute("""
            SELECT total_reviews, average_rating, satisfaction_percentage,
                   positive_count, neutral_count, negative_count, aspect_sentiment
            FROM app_metrics
            WHERE app_id = %s
        """, (app_id,))
        metrics = cursor.fetchone()

        if metrics:
            comparison_data.append({
                "app_id": app_id,
                "app_name": app_info["app_name"],
                "metrics": metrics
            })

    cursor.close()
    conn.close()

    if not comparison_data:
        return jsonify({"status": "fail", "message": "No data found for selected apps"}), 404

    return jsonify({
        "status": "success",
        "comparison_data": comparison_data
    })
    

if __name__ == "__main__":
    app.run(debug=True)

