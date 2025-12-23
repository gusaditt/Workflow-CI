import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

DATA_PATH = "dataset_preprocessing/dataset_preprocessing.csv"

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Log (TANPA start_run)
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, "model")

print("CI training selesai. Accuracy:", acc)