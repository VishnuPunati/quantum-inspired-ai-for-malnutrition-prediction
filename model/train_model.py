import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "health_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop("label", axis=1)
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
) 

X_train_q = X_train[:300]
y_train_q = y_train[:300]

X_test_q = X_test[:100]
y_test_q = y_test[:100]

feature_map = ZZFeatureMap(
    feature_dimension=4,
    reps=2,
    entanglement="full"
)

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

quantum_model = QSVC(
    quantum_kernel=quantum_kernel,
    C=1.5
)

print("Training Quantum Model...")
quantum_model.fit(X_train_q, y_train_q)

quantum_predictions = quantum_model.predict(X_test_q)
quantum_acc = accuracy_score(y_test_q, quantum_predictions)

print(f"Quantum Accuracy: {quantum_acc:.4f}")

classical_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

print("Training Classical Model...")
classical_model.fit(X_train, y_train)

classical_predictions = classical_model.predict(X_test)
classical_acc = accuracy_score(y_test, classical_predictions)

print(f"Classical Accuracy: {classical_acc:.4f}")

joblib.dump(quantum_model, os.path.join(MODEL_DIR, "quantum_model.pkl"))
joblib.dump(classical_model, os.path.join(MODEL_DIR, "classical_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("Models and scaler saved successfully")