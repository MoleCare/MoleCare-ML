from flask import Flask

app = Flask(__name__)

@app.route('/hello/', methods=['GET', 'POST'])
def home():
    return "Hello"
