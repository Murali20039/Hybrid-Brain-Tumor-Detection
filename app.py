import os
import requests
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json

app = Flask(__name__)
app.secret_key = "supersecret123"

# ------------------------------
# USER JSON FILE
# ------------------------------
USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        return {"users": []}
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(data):
    with open(USER_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ------------------------------
# MODEL LOADING
# ------------------------------
MODEL_PATH = "keras_model.h5"
model = load_model(MODEL_PATH)

class_names = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

# ------------------------------
# LOGIN CHECK
# ------------------------------
def login_required(func):
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login_page"))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


# ------------------------------
# Login Page
# ------------------------------
@app.route("/")
def login_page():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    email = request.form["email"]
    password = request.form["password"]

    users = load_users()["users"]

    for u in users:
        if u["email"] == email and u["password"] == password:
            session["user"] = email
            return redirect("/index")

    flash("Incorrect email or password!", "error")
    return redirect("/")


# ------------------------------
# Signup Page
# ------------------------------
@app.route("/signup")
def signup_page():
    return render_template("signup.html")


@app.route("/register", methods=["POST"])
def register():
    name = request.form["name"]
    email = request.form["email"]
    mobile = request.form["mobile"]
    password = request.form["password"]

    data = load_users()

    for u in data["users"]:
        if u["email"] == email:
            flash("User already exists!", "error")
            return redirect("/signup")

    data["users"].append({
        "name": name,
        "email": email,
        "mobile": mobile,
        "password": password
    })

    save_users(data)

    flash("Signup successful! Please login.", "success")
    return redirect("/")


# ------------------------------
# LOGOUT
# ------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ------------------------------
# HOME (Protected)
# ------------------------------
@app.route("/index")
@login_required
def index():
    return render_template("index.html")


# ------------------------------
# ORIGINAL PREDICTION LOGIC
# ------------------------------
def is_mri_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((150, 150))
    arr = np.array(img)
    colored_pixels = 0
    total_pixels = arr.shape[0] * arr.shape[1]

    for pixel in arr.reshape(-1, 3):
        r, g, b = pixel
        if abs(r - g) > 25 or abs(r - b) > 25 or abs(g - b) > 25:
            colored_pixels += 1

    color_ratio = (colored_pixels / total_pixels) * 100
    return color_ratio < 10


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


# ------------------------------
# AI GENERATION
# ------------------------------
def generate_description(tumor_type):
    prompt = (
        f"Explain {tumor_type} in simple medical language. "
        "Write 2-3 short sentences."
    )

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "MRI-AI-Project"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post(url, json=data, headers=headers)

    print("\n=== RAW ===")
    print(r.text)

    try:
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "AI explanation temporarily unavailable."


# ------------------------------
# PREDICT Route
# ------------------------------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No image selected"

    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    if not is_mri_image(file_path):
        return render_template(
            "index.html",
            prediction="Unknown Image – Please upload a proper Brain MRI image.",
            uploaded_image=file_path
        )

    img = preprocess_image(file_path)
    preds = model.predict(img)[0]

    result = class_names[np.argmax(preds)]
    ai_description = generate_description(result)

    return render_template(
        "index.html",
        prediction=result,
        ai_description=ai_description,
        uploaded_image=file_path
    )


if __name__ == "__main__":
    app.run(debug=True)
