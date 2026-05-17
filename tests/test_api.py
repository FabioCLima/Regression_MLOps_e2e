"""API endpoint tests using TestClient with injected fake artifacts."""


def test_root_returns_service_info(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "regression-mlops-e2e-api"
    assert data["status"] == "running"


def test_health_returns_healthy_when_artifacts_loaded(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["encoders_loaded"] is True
    assert data["model_version"] is not None
    assert data["n_features_expected"] == 8


def test_health_returns_503_when_no_artifacts(api_client_no_artifacts):
    response = api_client_no_artifacts.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False


def test_predict_happy_path_returns_price(api_client, minimal_predict_payload):
    response = api_client.post("/predict", json=minimal_predict_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
    assert data["predicted_price"] > 0


def test_predict_returns_model_version(api_client, minimal_predict_payload):
    response = api_client.post("/predict", json=minimal_predict_payload)
    assert response.status_code == 200
    assert response.json()["model_version"] == "fake_model@sha256:abcd1234@local"


def test_predict_returns_missing_features_when_optionals_absent(api_client, minimal_predict_payload):
    response = api_client.post("/predict", json=minimal_predict_payload)
    assert response.status_code == 200
    # With only required fields sent, many train_columns will be missing
    missing = response.json()["missing_features"]
    assert isinstance(missing, list)


def test_predict_rejects_missing_required_field(api_client):
    response = api_client.post("/predict", json={"date": "2022-01-01", "city_full": "Austin", "city": "AUS"})
    assert response.status_code == 422


def test_predict_rejects_unknown_field(api_client, minimal_predict_payload):
    payload = {**minimal_predict_payload, "unknown_column": 999}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_returns_503_without_artifacts(api_client_no_artifacts, minimal_predict_payload):
    response = api_client_no_artifacts.post("/predict", json=minimal_predict_payload)
    assert response.status_code == 503


def test_predict_response_includes_request_id_header(api_client, minimal_predict_payload):
    response = api_client.post("/predict", json=minimal_predict_payload)
    assert "x-request-id" in response.headers


def test_predict_propagates_provided_request_id(api_client, minimal_predict_payload):
    response = api_client.post(
        "/predict",
        json=minimal_predict_payload,
        headers={"X-Request-ID": "test-123"},
    )
    assert response.headers.get("x-request-id") == "test-123"


def test_predict_batch_returns_multiple_predictions(api_client, minimal_predict_payload):
    payload = {"records": [minimal_predict_payload, minimal_predict_payload]}
    response = api_client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rows_predicted"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert "predicted_price" in pred
        assert isinstance(pred["predicted_price"], float)


def test_predict_batch_rejects_empty_records(api_client):
    response = api_client.post("/predict/batch", json={"records": []})
    assert response.status_code == 422


def test_predict_batch_rejects_over_limit(api_client, minimal_predict_payload):
    payload = {"records": [minimal_predict_payload] * 1001}
    response = api_client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_predict_batch_returns_503_without_artifacts(api_client_no_artifacts, minimal_predict_payload):
    payload = {"records": [minimal_predict_payload]}
    response = api_client_no_artifacts.post("/predict/batch", json=payload)
    assert response.status_code == 503


def test_model_info_returns_metadata(api_client):
    response = api_client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "version_string" in data
    assert "n_features_expected" in data
    assert data["n_features_expected"] == 8
    assert isinstance(data["train_columns"], list)


def test_model_info_returns_503_without_artifacts(api_client_no_artifacts):
    response = api_client_no_artifacts.get("/model-info")
    assert response.status_code == 503
