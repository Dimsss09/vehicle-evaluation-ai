from app import create_app
from app.model_service import predict_car_safety


def test_model_predicts_known_payload():
    result = predict_car_safety(
        {
            "buying": "med",
            "maint": "med",
            "doors": "4",
            "persons": "4",
            "lug_boot": "med",
            "safety": "high",
        }
    )

    assert result["prediction"] in {"unacc", "acc", "good", "vgood"}
    assert result["confidence"] > 0


def test_homepage_loads():
    app = create_app()
    app.config.update(TESTING=True)

    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"Prediksi Keamanan Mobil" in response.data


def test_predict_rejects_invalid_input():
    app = create_app()
    app.config.update(TESTING=True)

    response = app.test_client().post(
        "/predict",
        data={
            "buying": "murah",
            "maint": "med",
            "doors": "4",
            "persons": "4",
            "lug_boot": "med",
            "safety": "high",
        },
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 400
