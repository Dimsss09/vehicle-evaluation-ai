from flask import Blueprint, jsonify, render_template, request

from .model_service import CLASS_LABELS, FEATURES, OPTIONS, get_model_bundle, predict_car_safety


bp = Blueprint("main", __name__)


@bp.get("/")
def index():
    _, metrics = get_model_bundle()
    defaults = {
        "buying": "med",
        "maint": "med",
        "doors": "4",
        "persons": "4",
        "lug_boot": "med",
        "safety": "high",
    }
    result = predict_car_safety(defaults)
    return render_template(
        "index.html",
        options=OPTIONS,
        features=FEATURES,
        class_labels=CLASS_LABELS,
        metrics=metrics,
        result=result,
        current_values=defaults,
    )


@bp.post("/predict")
def predict():
    form_values = {feature: request.form.get(feature, "") for feature in FEATURES}
    invalid = {
        key: value
        for key, value in form_values.items()
        if value not in OPTIONS.get(key, [])
    }
    if invalid:
        return jsonify({"error": "Input tidak valid", "invalid": invalid}), 400

    result = predict_car_safety(form_values)
    if request.headers.get("Accept", "").startswith("application/json"):
        return jsonify(result)

    _, metrics = get_model_bundle()
    return render_template(
        "index.html",
        options=OPTIONS,
        features=FEATURES,
        class_labels=CLASS_LABELS,
        metrics=metrics,
        result=result,
        current_values=form_values,
    )
