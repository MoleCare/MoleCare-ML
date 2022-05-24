from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def helloWorld():
    return "Hello, cross-origin-world!"

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/v1/users")
def list_users():
    return "user example"

@app.route("/")
@cross_origin()
def helloWorld():
    return "Hello, cross-origin-world!"


