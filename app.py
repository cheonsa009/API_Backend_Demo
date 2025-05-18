from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return "Server is running!"

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message') if data else ''
    
    if user_input.lower() == "hi":
        reply = "hello"
    else:
        reply = "I don't understand."
    
    return jsonify({"response": reply})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
