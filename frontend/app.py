import requests
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from _mlflow.registry import MLflowRegistry
from dotenv import load_dotenv

load_dotenv()

KSERVE_URL = os.environ.get("KSERVE_URL", "http://localhost:7070/v1/models/employee_attrition_prediction:predict")
MLFLOW_TRACKING_URI = "http://localhost:5000"


app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print('incoming-data: ', data)

    registry = MLflowRegistry(
        tracking_uri="http://localhost:5000",
        experiment_name="employee-attrition-v1"
    )

    try:
        features = registry.load_features_from_mlflow()
        missing = set(features) - set(data.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        df_input = pd.DataFrame([[data.get(f) for f in features]], columns=features)

        response = requests.post(
            KSERVE_URL, 
            json={"instances": df_input.to_dict(orient="records")}
        )
        print('results: ', response.json())

        prediction_result = response.json()
        
        payload = { 
            "prediction": prediction_result['prediction'][0], 
            "probs": prediction_result['probs'][0], 
        }

        return payload
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="localhost", port=4000, debug=True)