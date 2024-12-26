'''from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib  # For saving the model

# Load dataset
data = pd.read_csv("data.csv")

# Preprocess data
X = data.iloc[:, [4, 5, 6, 9, 10, 11]]
Y = data.iloc[:, 12]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

print("Model trained successfully!")
# Save the model to a file
joblib.dump(classifier, "groundwater_model.pkl")

# Save the scaler (optional, useful for future predictions)
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved as groundwater_model.pkl and scaler.pkl in the ml project folder!")'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, mean_squared_error

# Load dataset
data = pd.read_csv("C:/Users/Rajshekar/Downloads/data.csv")

# Encode "States" column
label_encoder = LabelEncoder()
data['States'] = label_encoder.fit_transform(data['States'])

# One-Hot Encoding for "Situation"
data = pd.get_dummies(data, columns=['Situation'], drop_first=True)

# Bin the target variable into categories
bins = [-np.inf, 5, 15, np.inf]  # Define bins for "Low", "Moderate", "High"
labels = ['Low', 'Moderate', 'High']  # Labels for the bins
data['Groundwater availability for future irrigation use'] = pd.cut(
    data['Groundwater availability for future irrigation use'],
    bins=bins,
    labels=labels
)

# Split data into features and target
X = data.drop(columns=['Groundwater availability for future irrigation use'])
y = data['Groundwater availability for future irrigation use']

# Encode target labels
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Logistic Regression
logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(X_train, y_train)
logistic_preds = logistic_model.predict(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Metrics
logistic_accuracy = accuracy_score(y_test, logistic_preds)
logistic_precision = precision_score(y_test, logistic_preds, average='weighted')
logistic_cm = confusion_matrix(y_test, logistic_preds)
logistic_mse = mean_squared_error(y_test, logistic_preds)

rf_accuracy = accuracy_score(y_test, rf_preds)
rf_precision = precision_score(y_test, rf_preds, average='weighted')
rf_cm = confusion_matrix(y_test, rf_preds)
rf_mse = mean_squared_error(y_test, rf_preds)

# Display Results
print("Logistic Regression:")
print(f"Accuracy: {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Confusion Matrix:\n{logistic_cm}")
print(f"MSE: {logistic_mse:.4f}")

print("\nRandom Forest:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Confusion Matrix:\n{rf_cm}")
print(f"MSE: {rf_mse:.4f}")