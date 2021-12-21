#!/bin/sh
# makes it executable: >chmod +x bootstrap.sh

export FLASK_APP=app.py
#source $(pipenv --venv)/bin/activate
python3 -m flask run -h 0.0.0.0