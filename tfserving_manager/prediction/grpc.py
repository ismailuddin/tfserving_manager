"""Class for getting predictions from the TensorFlow Serving server
using the gRPC API"""


import numpy as np
import grpc
from typing import List, Tuple
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.prediction_service_pb2_grpc import (
    PredictionServiceStub,
)
import tensorflow as tf


class GRPCPredictionAPI:
    """Class for interacting with TensorFlow Serving server using gRPC"""

    def __init__(self, host: str = "localhost", port: int = 8500):
        self.host = host
        self.port = port
        self.url = f"{self.host}:{self.port}"
        channel = grpc.insecure_channel(self.url)
        self.stub = PredictionServiceStub(channel)

    def get_prediction(
        self,
        model_name: str,
        model_version: int,
        inputs: np.ndarray,
        input_layer_name: str,
        output_layer_name: str,
        input_shape: Tuple[int],
        output_shape: Tuple[int] = None
    ) -> np.ndarray:
        """Get predictions from TensorFlow Serving server, from the specified
        model, version and input.

        Args:
            model_name (str): Model name
            model_version (int): Version of model
            inputs (np.ndarray): Input as a NumPy array, in the correct shape
                as expected by the model. This may require an extra axis for
                number of instances of the input e.g. (1, 224, 224, 3)
            input_layer_name (str): Input layer name in model
            output_layer_name (str): Output layer in model
            input_shape (Tuple[int]): Shape of the input. Depending on the
                model, an extra first axis may be required which encodes
                the number of instances of the input e.g. (1, 224, 224, 3)
            output_shape (Tuple[int]): Shape of the model output, where
                typically the first axis is the number of instances of the
                input provided.

        Returns:
            np.ndarray: Predictions from model
        """
        request = PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = "serving_default"
        request.inputs[input_layer_name].CopyFrom(
            tf.make_tensor_proto(
                inputs.astype(np.float32), shape=input_shape
            )
        )
        result = self.stub.Predict(request)
        return np.array(result.outputs[output_layer_name].float_val).reshape(output_shape)
