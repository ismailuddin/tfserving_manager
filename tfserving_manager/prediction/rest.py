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
            inputs (List[List[float]]): Inputs to the model, in the correct
                shape expected by the model, but as a Python list. This is
                typically the equivalent of a NumPy array of shape for example
                of (1, 224, 224, 3), where 1 is the number of images and each
                image is of shape (224, 224, 3). Use NumPy's `.tolist()` to
                convert each array into a list, and place inside another Python
                list to achieve the correct dimensionality.

        Returns:
            np.ndarray: Predictions from model
        """
        response = requests.post(
            f"{self.url}/v{model_version}/models/{model_name}:predict",
            json={
                "signature_name": "serving_default",
                "instances": inputs,
            },
            headers={"Content-Type": "application/json"}
        )
        return np.array(json.loads(response.content.decode())["predictions"])
