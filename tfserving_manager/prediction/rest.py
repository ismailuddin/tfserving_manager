"""Class for getting predictions from the TensorFlow Serving server
using the RESTful API endpoints"""


import numpy as np
import requests
import json


class RESTAPIPrediction:
    """Class for getting predictions from the TensorFlow Serving server using
    the RESTful API enpdoints"""

    def __init__(self, host: str = "localhost", port: int = 8501):
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}"

    def get_prediction(
        self, model_name: str, model_version: int, inputs: np.ndarray
    ) -> np.ndarray:
        """Get predictions from TensorFlow Serving server, from the specified
        model, version and input.

        Args:
            model_name (str): Model name
            model_version (int): Version of model
            inputs (np.ndarray): Input as a NumPy array, exluding dimension for
                batch support

        Returns:
            np.ndarray: Predictions from model
        """
        response = requests.post(
            f"{self.url}/v{model_version}/models/{model_name}:predict",
            json={
                "signature_name": "serving_default",
                "instances": [inputs.tolist()],
            },
            headers={"Content-Type": "application/json"}
        )
        return np.array(json.loads(response.content.decode())["predictions"])
