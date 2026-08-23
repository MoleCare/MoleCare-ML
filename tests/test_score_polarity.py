"""Regression tests for the score-polarity defect (audit F1).

The model was trained with {'Melanoma': 0, 'NotMelanoma': 1}, so its raw output
is P(NotMelanoma). Two call sites inverted it and two did not, so /predict and
/predict-advanced reported opposite things about the same image, and the Lambda
entrypoint agreed with the wrong one.

These tests pin the polarity in one place. If someone reads prediction[0][0]
directly again, the agreement test below fails.
"""

import pytest

from ml_model_serving.model_prediction_service import (
    melanoma_probability,
    not_melanoma_percent,
)

# --------------------------------------------------------------------------
# The helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_melanoma_prob",
    [
        (0.0, 1.0),   # raw 0 == class Melanoma  -> certainty of melanoma
        (1.0, 0.0),   # raw 1 == class NotMelanoma
        (0.87, 0.13),
        (0.5, 0.5),
    ],
)
def test_melanoma_probability_inverts_the_raw_score(raw, expected_melanoma_prob):
    assert melanoma_probability(raw) == pytest.approx(expected_melanoma_prob)


def test_a_high_raw_score_means_low_melanoma_risk():
    """The whole defect in one assertion."""
    assert melanoma_probability(0.95) < 0.1


def test_legacy_percent_keeps_its_documented_meaning():
    """`percent` is P(NOT melanoma) as 0-100 -- molecare-server persists it.

    Changing this silently would corrupt stored predictions in the Java backend,
    so the meaning is pinned until that consumer migrates.
    """
    assert not_melanoma_percent(0.87) == pytest.approx(87.0)


def test_percent_and_melanoma_probability_are_complements():
    for raw in (0.0, 0.25, 0.5, 0.87, 1.0):
        assert not_melanoma_percent(raw) / 100 + melanoma_probability(raw) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The endpoints must agree
# --------------------------------------------------------------------------

def test_predict_reports_melanoma_probability(client, predict_body, fake_model):
    data = client.post("/predict", json=predict_body).get_json()["data"]
    assert "melanomaProbability" in data, "/predict must expose P(melanoma)"
    assert data["melanomaProbability"] == pytest.approx(1 - fake_model.raw, abs=1e-6)


def test_predict_still_serves_the_deprecated_percent_field(client, predict_body, fake_model):
    """molecare-server reads this; removing it is a breaking cross-repo change."""
    data = client.post("/predict", json=predict_body).get_json()["data"]
    assert data["percent"] == pytest.approx(fake_model.raw * 100, abs=1e-4)


def test_predict_and_percent_do_not_contradict_each_other(client, predict_body):
    """The exact contradiction the audit found: percent 95 vs probability 0.05."""
    data = client.post("/predict", json=predict_body).get_json()["data"]
    assert data["percent"] / 100 + data["melanomaProbability"] == pytest.approx(1.0, abs=1e-6)


def test_lambda_handler_agrees_with_the_flask_endpoint(fake_model, predict_body, monkeypatch):
    """handler.py is the production path and previously used the wrong polarity."""
    import handler

    monkeypatch.setattr(handler, "model_service", None, raising=False)
    handler._initialize()

    event = {"body": __import__("json").dumps(predict_body)}
    response = handler.predict(event, None)
    body = __import__("json").loads(response["body"])
    data = body["data"] if "data" in body else body

    assert data["melanomaProbability"] == pytest.approx(1 - fake_model.raw, abs=1e-6)
    assert data["percent"] / 100 + data["melanomaProbability"] == pytest.approx(1.0, abs=1e-6)
