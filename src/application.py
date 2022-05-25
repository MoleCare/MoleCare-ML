from flask import Flask

from .containers import Container
from . import controller


def create_app() -> Flask:
    container = Container()

    app = Flask(__name__)
    app.container = container
    app.add_url_rule("/c1", "index", controller.api)
    app.add_url_rule("/c2", "index", controller.api)

    return app