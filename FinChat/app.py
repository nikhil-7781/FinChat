from flask import Flask, request, jsonify, render_template
from setup import backend_bot_llm
app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    prompt=request.json.get("message", "")
    response=backend_bot_llm(prompt)
    return response

if __name__=="__main__":
    app.run(debug=True)