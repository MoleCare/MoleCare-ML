#!/bin/sh
export FLASK_APP=./app.py
export FLASK_ENV=development
python app.py
flask run -h 0.0.0.0 $PORT
