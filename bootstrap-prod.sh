#!/bin/sh
export FLASK_APP=./src/application.py
export FLASK_ENV=production
source $(pipenv --venv)/bin/activate
flask run -h 0.0.0.0
#waitress-serve --call 'flaskr:create_app'