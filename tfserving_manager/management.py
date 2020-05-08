import grpc
from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub
from tensorflow_serving.config.model_server_config_pb2 import ModelServerConfig
from tensorflow_serving.apis.model_management_pb2 import ReloadConfigRequest


class GRPCModelServiceAPI:
    def __init__(self, host: str = "localhost", port: int = 8500):
        self.host = host
        self.port = port
        self.url = f"{self.host}:{self.port}"
        channel = grpc.insecure_channel(self.url)
        self.stub = ModelServiceStub(channel)

    def replace_server_config(self, model_config: ModelServerConfig):
        self.stub.HandleReloadConfigRequest(ReloadConfigRequest(
            config=model_config
        ))
