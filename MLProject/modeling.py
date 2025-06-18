import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

# Set MLflow experiment
mlflow.set_experiment("Sleep_Quality_Prediction - Yahya")

# Load preprocessed dataset
df = pd.read_csv('namadataset_preprocessing\sleep_data_processed.csv')

X = df.drop('Sleep Quality', axis=1)
y = df['Sleep Quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable autologging
mlflow.sklearn.autolog()

# Train model
with mlflow.start_run(run_name="baseline_random_forest"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Log metrics manually for clarity
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")