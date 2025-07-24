# Importing required libraries
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import warnings
import joblib  # Better than pickle for sklearn models
from sklearn import metrics
from feature import FeatureExtraction

# Suppress warnings
warnings.filterwarnings('ignore')

# Load model
try:
    gbc = joblib.load("pickle/models.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# Initialize Flask app
app = Flask(__name__)

# Home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url", "")
        
        if not url:
            return render_template("index.html", xx=-1, error="No URL provided.")
        
        try:
            # Feature extraction
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1, 30)

            # Predictions
            y_pred = gbc.predict(x)[0]
            y_pro_phishing = gbc.predict_proba(x)[0, 0]
            y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

            pred = "It is {0:.2f}% safe to go".format(y_pro_phishing * 100)
            return render_template("index.html", xx=round(y_pro_non_phishing, 2), url=url)

        except Exception as e:
            return render_template("index.html", xx=-1, error=str(e))

    return render_template("index.html", xx=-1)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
