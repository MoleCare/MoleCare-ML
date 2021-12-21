from flask import Flask, render_template, request, url_for, jsonify,Response, jsonify

app = Flask(__name__)

@app.route('/hello/', methods=['GET', 'POST'])
def home():
    return "Hello"
