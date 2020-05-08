import numpy as np
import requests
import json


class RESTAPIPrediction:
    def __init__(self, host: str = "localhost", port: int = 8501):
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}"

    def get_prediction(
        self, model_name: str, model_version: int, inputs: np.ndarray
    ) -> np.ndarray:
        response = requests.post(
            f"{self.url}/v{model_version}/models/{model_name}:predict",
            json={
                "signature_name": "serving_default",
                "instances": [inputs.tolist()],
            },
            headers={"Content-Type": "application/json"}
        )
        return np.array(json.loads(response.content.decode())["predictions"])
