from flask import Flask, render_template, request
import re

#  AraBERT imports - 
import tensorflow as tf
from transformers import AutoTokenizer

app = Flask(__name__)

# =========================
# Load things ONCE at startup
# =========================
print("Loading Model/Tokenizer...")

# AraBERT 
# tokenizer = AutoTokenizer.from_pretrained("aub-mind/bert-base-arabertv02-twitter")
# model = tf.keras.models.load_model("models/arabert_saved.h5")

print("Loaded successfully.")

# =========================
# Preprocessing (مؤقت) -
# =========================
def clean_tweet(text: str) -> str:
    text = text.strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)                  # mentions
    text = re.sub(r"#\w+", "", text)                  # hashtags
    text = re.sub(r"\s+", " ", text)                  # multi-spaces
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    original_text = ""
    cleaned_text = ""
    error = None

    if request.method == "POST":
        original_text = request.form.get("tweet", "")

        # 1) Empty check
        if not original_text.strip():
            error = "المرجو كتابة نص قبل الضغط على تحليل."
            return render_template(
                "index.html",
                prediction=prediction,
                text=original_text,
                cleaned_text=cleaned_text,
                error=error
            )

        # 2) Clean
        cleaned_text = clean_tweet(original_text)

        # 3) Predict (Placeholder until model is ready)
        # ===== ملي يكون AraBERT واجد =====
        # inputs = tokenizer(cleaned_text, return_tensors="tf", padding=True, truncation=True)
        # probs = model.predict(dict(inputs))  # حسب شكل model عندك
        # pred = int(probs.argmax(axis=1)[0])
        # prediction = "Hate Speech" if pred == 1 else "Normal"

        prediction = "Normal"  # 

    return render_template(
        "index.html",
        prediction=prediction,
        text=original_text,
        cleaned_text=cleaned_text,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
