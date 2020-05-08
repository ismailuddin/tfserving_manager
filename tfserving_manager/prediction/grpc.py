import numpy as np
import grpc
from typing import List, Tuple
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.prediction_service_pb2_grpc import (
    PredictionServiceStub,
)
import tensorflow as tf


class GRPCPredictionAPI:
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
    ) -> np.ndarray:
        request = PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = "serving_default"
        request.inputs[input_layer_name].CopyFrom(
            tf.make_tensor_proto(
                inputs[np.newaxis].astype(np.float32), shape=input_shape
            )
        )
        result = self.stub.Predict(request)
        return np.array(result.outputs[output_layer_name].float_val)
