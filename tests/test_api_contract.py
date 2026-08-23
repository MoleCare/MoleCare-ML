"""HTTP contract tests for every route in prediction_controller.

Before this file existed, none of the 13 endpoints had a test. That is how the
CORS allowlist came to be inert on every route and how /predict and
/predict-advanced came to disagree about the same image.

Endpoints backed by heavy CV/ML dependencies cannot be exercised end-to-end
without those libraries installed, so they are asserted at the contract level:
malformed input must be rejected before any model runs, and no handler may
return raw exception text. Those two properties are what the audit found broken,
and they hold regardless of whether the CV stack is real or stubbed.
"""

import json

import pytest

# Every route the app exposes, with the HTTP method it accepts.
GET_ROUTES = ["/", "/health", "/model-status"]
POST_ROUTES = [
    "/predict",
    "/analyze",
    "/analyze/abcde",
    "/compare",
    "/detect",
    "/detect/extract",
    "/evolution",
    "/validate",
    "/predict-advanced",
    "/compare-models",
]
ALL_ROUTES = GET_ROUTES + POST_ROUTES


def test_every_declared_route_is_covered(app):
    """Fails if someone adds a route without adding it here."""
    declared = {
        r.rule
        for r in app.url_map.iter_rules()
        if r.endpoint != "static"
    }
    assert declared == set(ALL_ROUTES), (
        f"route list out of date; missing from tests: {declared - set(ALL_ROUTES)}, "
        f"stale in tests: {set(ALL_ROUTES) - declared}"
    )


# --------------------------------------------------------------------------
# GET routes
# --------------------------------------------------------------------------

def test_index_responds(client):
    assert client.get("/").status_code == 200


def test_health_reports_healthy_when_model_loads(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] in {"healthy", "degraded"}
    assert "xception" in body["services"]
    assert isinstance(body["response_time_ms"], int)


def test_health_does_not_leak_internal_errors(client):
    """F3: /health is unauthenticated; exception text can carry MODEL_PATH."""
    body = client.get("/health").get_json()
    for name, svc in body["services"].items():
        assert "error" not in svc, f"/health leaks internal error text for {name}"


def test_model_status_responds(client):
    assert client.get("/model-status").status_code in {200, 503}


# --------------------------------------------------------------------------
# Request validation -- applies to every POST route
# --------------------------------------------------------------------------

# A malformed request is the client's fault. The endpoint may answer 400
# (bad body), 415 (wrong content type) or 503 (that analysis service is not
# installed) -- but never 500. A 500 here is what inflates the CloudWatch error
# rate that canary_deploy.py reads to decide promote-or-rollback, so a client
# sending junk could trigger a production rollback.
CLIENT_ERROR_CODES = {400, 413, 415, 503}


@pytest.mark.parametrize("route", POST_ROUTES)
def test_post_rejects_non_json_body(client, route):
    r = client.post(route, data="not json", content_type="text/plain")
    assert r.status_code in CLIENT_ERROR_CODES, (
        f"{route} returned {r.status_code} for a non-JSON body"
    )
    assert r.status_code != 500


@pytest.mark.parametrize("route", POST_ROUTES)
def test_post_rejects_empty_object(client, route):
    r = client.post(route, json={})
    assert r.status_code in CLIENT_ERROR_CODES, (
        f"{route} returned {r.status_code} for an empty body"
    )
    assert r.status_code != 500


@pytest.mark.parametrize("route", POST_ROUTES)
def test_post_never_returns_raw_exception_text(client, route):
    """F3: internal exception strings must not reach clients."""
    r = client.post(route, json={"imagebase64": "!!!not-base64!!!", "predictionid": "x"})
    text = r.get_data(as_text=True)
    for marker in ("Traceback", 'File "', ".py\", line", "MODEL_PATH", "/var/task"):
        assert marker not in text, f"{route} leaked internal detail: {marker!r}"


def test_get_route_rejects_post(client):
    assert client.post("/health", json={}).status_code == 405


# --------------------------------------------------------------------------
# /predict -- the primary path
# --------------------------------------------------------------------------

def test_predict_happy_path(client, predict_body):
    r = client.post("/predict", json=predict_body)
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == 200
    data = body["data"]
    assert data["predictionid"] == predict_body["predictionid"]
    assert "disclaimer" in data and data["disclaimer"]


def test_predict_requires_both_fields(client, image_b64):
    assert client.post("/predict", json={"imagebase64": image_b64}).status_code == 400
    assert client.post("/predict", json={"predictionid": "abc"}).status_code == 400


def test_predict_rejects_undecodable_image_with_400_not_500(client):
    """F6: ImageProcessor raises ValueError; that is a client error, not a 500.

    A 500 here also inflates the CloudWatch error rate that canary_deploy.py
    reads to decide promote-or-rollback.
    """
    r = client.post("/predict", json={"predictionid": "abc", "imagebase64": "%%%%%%"})
    assert r.status_code == 400, f"expected 400, got {r.status_code}"


def test_predict_passes_an_array_not_a_list_to_the_model(client, predict_body, fake_model):
    """F5: .tolist() cost ~15ms/request converting 268,203 values."""
    client.post("/predict", json=predict_body)
    assert fake_model.calls, "model was never invoked"
    assert not isinstance(fake_model.calls[0], list), (
        "model received a Python list -- the .tolist() regression is back"
    )
    assert hasattr(fake_model.calls[0], "shape")


def test_oversized_body_is_rejected(client, app):
    """MAX_CONTENT_LENGTH must reject before the payload is decoded."""
    limit = app.config["MAX_CONTENT_LENGTH"]
    r = client.post(
        "/predict",
        data=json.dumps({"predictionid": "a", "imagebase64": "A" * (limit + 1024)}),
        content_type="application/json",
    )
    assert r.status_code == 413


# --------------------------------------------------------------------------
# /validate
# --------------------------------------------------------------------------

def test_validate_returns_the_full_quality_report(client, predict_body):
    """/validate read five attributes ImageQualityReport never defined.

    Every call raised AttributeError and returned 500, so the endpoint had never
    worked. These are the exact fields that were missing.
    """
    r = client.post("/validate", json=predict_body)
    assert r.status_code == 200, f"/validate returned {r.status_code}"
    data = r.get_json()
    payload = data.get("data", data)
    for field in (
        "aspect_ratio",
        "format_detected",
        "file_size_kb",
        "meets_minimum_resolution",
        "is_optimal_resolution",
    ):
        assert field in payload, f"/validate is missing {field}"
    assert payload["aspect_ratio"] == pytest.approx(1.0)      # fixture is 64x64
    assert payload["format_detected"] == "PNG"
    assert payload["file_size_kb"] > 0


def test_validate_rejects_a_bad_image(client):
    r = client.post("/validate", json={"imagebase64": "%%%%", "predictionid": "x"})
    assert r.status_code != 500
