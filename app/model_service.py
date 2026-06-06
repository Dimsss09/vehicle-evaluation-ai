from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATASET_PATH = DATA_DIR / "car.data"
MODEL_PATH = MODEL_DIR / "car_safety_random_forest.joblib"
METRICS_PATH = MODEL_DIR / "metrics.joblib"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

FEATURES = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
TARGET = "class"
CLASS_LABELS = {
    "unacc": {
        "title": "Tidak Layak",
        "tone": "danger",
        "description": "Kombinasi fitur dinilai belum aman atau belum layak diterima.",
    },
    "acc": {
        "title": "Layak",
        "tone": "warning",
        "description": "Mobil dapat diterima, tetapi masih ada beberapa faktor yang perlu diperhatikan.",
    },
    "good": {
        "title": "Baik",
        "tone": "good",
        "description": "Mobil berada pada kategori baik berdasarkan biaya, kapasitas, bagasi, dan keselamatan.",
    },
    "vgood": {
        "title": "Sangat Baik",
        "tone": "excellent",
        "description": "Mobil memiliki kombinasi fitur yang sangat kuat untuk aspek keamanan dan kelayakan.",
    },
}

OPTIONS = {
    "buying": ["vhigh", "high", "med", "low"],
    "maint": ["vhigh", "high", "med", "low"],
    "doors": ["2", "3", "4", "5more"],
    "persons": ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety": ["low", "med", "high"],
}


def ensure_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if DATASET_PATH.exists():
        return DATASET_PATH

    df = pd.read_csv(DATASET_URL, header=None)
    df.to_csv(DATASET_PATH, header=False, index=False)
    return DATASET_PATH


def load_dataset():
    ensure_dataset()
    columns = FEATURES + [TARGET]
    return pd.read_csv(DATASET_PATH, names=columns)


def train_and_save_model():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    encoder = ColumnTransformer(
        transformers=[
            (
                "ordinal_features",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                FEATURES,
            )
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("encoder", encoder),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=0)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    classifier = pipeline.named_steps["classifier"]
    importances = dict(
        sorted(
            zip(FEATURES, classifier.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=["acc", "good", "unacc", "vgood"]
        ).tolist(),
        "class_order": ["acc", "good", "unacc", "vgood"],
        "feature_importances": importances,
        "dataset_rows": len(df),
        "test_rows": len(y_test),
    }
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    return pipeline, metrics


def get_model_bundle():
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(METRICS_PATH)
    return train_and_save_model()


def predict_car_safety(payload):
    model, metrics = get_model_bundle()
    row = {feature: payload[feature] for feature in FEATURES}
    sample = pd.DataFrame([row], columns=FEATURES)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    classes = model.named_steps["classifier"].classes_
    probability_map = {
        cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)
    }
    return {
        "input": row,
        "prediction": prediction,
        "label": CLASS_LABELS[prediction],
        "probabilities": probability_map,
        "confidence": probability_map[prediction],
        "metrics": metrics,
    }
