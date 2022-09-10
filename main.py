from flask import Flask, jsonify, request
from helper import set_embedder
from generator import SummaryGeneratorExtractive
from flask_cors import CORS
import constants

app = Flask(__name__)
app.config["SECRET_KEY"] = constants.SECRET_KEY
set_embedder()
CORS(app)

obj = SummaryGeneratorExtractive()
app = Flask(__name__)


@app.route('/get_summary', methods=["POST"])
def get_summary():
    text = request.json.get("text").strip().lower()
    result = obj.get_summary_result(text)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
