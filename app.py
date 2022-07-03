from flask import Flask, redirect, url_for
from flask_cors import CORS, cross_origin
import os
import re
from datetime import datetime

app = Flask(__name__)

#cross domain requests
CORS(app)

@app.route('/')
@cross_origin()
def index():
    return redirect(url_for('home'))

@app.route('/hello/', methods=['GET', 'POST'])
@cross_origin()
def home():
    return "Hello"

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + \
              "! It's " + formatted_now
    return content


if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get('PORT', 5000))
    #serve(app, host="0.0.0.0", port=port)
    app.run(host='0.0.0.0', port=port, debug=True)