import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("data.csv")

# 2️⃣ Encode categorical columns
le = LabelEncoder()
for col in ["Name", "Gender", "Degree", "Domain", "Skills", "PreferredRole"]:
    df[col] = le.fit_transform(df[col])

# Save LabelEncoder for decoding PreferredRole
role_le = LabelEncoder()
role_le.fit(df['PreferredRole'])
joblib.dump(role_le, "role_encoder.pkl")

# 3️⃣ Split features and target
X = df.drop("PreferredRole", axis=1)
y = df["PreferredRole"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Save the trained model
joblib.dump(model, "career_model.pkl")

print(" Model and encoder saved successfully!")
