# model.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("liver.csv", encoding='latin1')  # <-- Replace filename if needed

# Basic preprocessing
data.columns = data.columns.str.strip()  # Remove any column space issues

# Drop rows with missing values (optional: you could also fillna)
data = data.dropna()

# Encode Gender
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Features and Target
X = data.drop(columns=['Result'])  # Assuming 'Result' is your target (1/0)
y = data['Result']

# Feature order (make sure this matches the columns in your app)
FEATURE_ORDER = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Save model, scaler, and feature order
pickle.dump(model, open("liver_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(FEATURE_ORDER, open("feature_order.pkl", "wb"))
