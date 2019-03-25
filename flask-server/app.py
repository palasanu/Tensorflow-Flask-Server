from flask import request, url_for, Flask, jsonify
from flask_api import FlaskAPI, status, exceptions
import webclassifier
import numpy as np

app = Flask(__name__)

labels = ['label1', 'label2', 'label3', 'label4', 'label5']

@app.route("/getPrediction", methods=['POST'])
def get_prediction():

    site_data = request.get_json()

    keywords = site_data['metaKeywords']
    description = site_data['metaDescription']
    content = site_data['content']

    processed_keywords, processed_description, processed_content = webclassifier.process(keywords, description, content)

    predictions = webclassifier.predict(processed_keywords , processed_description, processed_content)

    print(predictions)

    json_predictions = list()

    for conf_value, label in zip(predictions, labels):
        json_prediction = dict()
        json_prediction["labelName"] = label
        json_prediction["probability"] = str(round(conf_value, 2))
        json_predictions.append(json_prediction)

    return jsonify(json_predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8600, debug=True)