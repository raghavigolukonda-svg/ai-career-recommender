from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained model and LabelEncoder
model = joblib.load("career_model.pkl")
role_le = joblib.load("role_encoder.pkl")
le = LabelEncoder()  # For encoding input features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    name = request.form['name']
    gender = request.form['gender']
    degree = request.form['degree']
    domain = request.form['domain']
    skills = request.form['skills']
    score = int(request.form['score'])

    # Prepare input
    sample = pd.DataFrame([{
        "Name": name,
        "Gender": gender,
        "Degree": degree,
        "Domain": domain,
        "Skills": skills,
        "Score": score
    }])

    # Encode input columns
    for col in ["Name", "Gender", "Degree", "Domain", "Skills"]:
        sample[col] = le.fit_transform(sample[col])

    # Align columns
    sample = sample.reindex(columns=['Name','Gender','Degree','Domain','Skills','Score'], fill_value=0)

    # Predict
    prediction = model.predict(sample)[0]

    # Decode numeric code to role name
    predicted_role = role_le.inverse_transform([prediction])[0]

    return render_template('index.html', prediction_text=f"🎓 Recommended Career Role: {predicted_role}")

if __name__ == '__main__':
    app.run(debug=True)
