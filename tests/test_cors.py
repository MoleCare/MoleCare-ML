"""Regression tests for the CORS allowlist (audit F2).

An allowlist was added to the app, but every route also carried a bare
`@cross_origin()`. That decorator defaults to origins='*' and overrides the
app-level configuration, so the allowlist had no effect on any of the 13 routes
and any website could call this inference API from a visitor's browser.

These tests assert the behaviour, not the implementation, so they stay valid
whichever way the allowlist is expressed.
"""

import pytest

ALLOWED = "http://localhost:3000"
FOREIGN = "https://evil.example.com"

ROUTES = ["/", "/health", "/model-status"]


@pytest.mark.parametrize("route", ROUTES)
def test_allowed_origin_is_echoed_back(client, route):
    r = client.get(route, headers={"Origin": ALLOWED})
    assert r.headers.get("Access-Control-Allow-Origin") == ALLOWED


@pytest.mark.parametrize("route", ROUTES)
def test_foreign_origin_is_refused(client, route):
    r = client.get(route, headers={"Origin": FOREIGN})
    acao = r.headers.get("Access-Control-Allow-Origin")
    assert acao != FOREIGN, f"{route} reflected a foreign origin back to the browser"
    assert acao != "*", f"{route} answered with a wildcard origin"


def test_predict_refuses_a_foreign_origin(client, predict_body):
    r = client.post("/predict", json=predict_body, headers={"Origin": FOREIGN})
    acao = r.headers.get("Access-Control-Allow-Origin")
    assert acao not in (FOREIGN, "*")


def test_preflight_from_a_foreign_origin_is_refused(client):
    r = client.options(
        "/predict",
        headers={
            "Origin": FOREIGN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.headers.get("Access-Control-Allow-Origin") not in (FOREIGN, "*")


def test_no_route_reintroduces_a_bare_cross_origin_decorator():
    """The defect was a decorator silently overriding app config.

    Asserted at source level because a bare @cross_origin() on a route that
    happens to be otherwise untested would not show up in the behaviour tests.
    """
    import pathlib
    import re

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "ml_model_serving"
        / "prediction_controller.py"
    ).read_text()

    # Ignore the warning comment that explains why this must not come back.
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"^\s*@cross_origin\(\s*\)", code, re.M), (
        "a bare @cross_origin() is back; it defaults to origins='*' and "
        "overrides the allowlist"
    )
