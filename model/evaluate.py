import os
import joblib
import json

from .metrics import classification_metrics, measure_inference_time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved")
REPORT_DIR = os.path.join(BASE_DIR, "..", "reports")

os.makedirs(REPORT_DIR, exist_ok=True)


quantum_model = joblib.load(os.path.join(MODEL_DIR, "quantum_model.pkl"))
classical_model = joblib.load(os.path.join(MODEL_DIR, "classical_model.pkl"))

X_test = joblib.load(os.path.join(MODEL_DIR, "X_test.pkl"))
y_test = joblib.load(os.path.join(MODEL_DIR, "y_test.pkl"))

X_test_q = joblib.load(os.path.join(MODEL_DIR, "X_test_q.pkl"))
y_test_q = joblib.load(os.path.join(MODEL_DIR, "y_test_q.pkl"))


q_pred = quantum_model.predict(X_test_q)
c_pred = classical_model.predict(X_test)


q_metrics = classification_metrics(y_test_q, q_pred)
c_metrics = classification_metrics(y_test, c_pred)


q_time = measure_inference_time(quantum_model, X_test_q)
c_time = measure_inference_time(classical_model, X_test)


print("Quantum Model Metrics:", q_metrics)
print("Classical Model Metrics:", c_metrics)

print("Quantum inference time:", q_time)
print("Classical inference time:", c_time)


with open(os.path.join(REPORT_DIR, "model_metrics.json"), "w") as f:
    json.dump(
        {
            "quantum_model": q_metrics,
            "classical_model": c_metrics,
            "quantum_inference_time": q_time,
            "classical_inference_time": c_time
        },
        f,
        indent=4
    )