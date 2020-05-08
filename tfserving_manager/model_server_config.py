import json
from typing import List
from tensorflow_serving.config import model_server_config_pb2
from tensorflow_serving.config.model_server_config_pb2 import ModelServerConfig
from google.protobuf import text_format, json_format
from model_server.exceptions import ModelConfigError


class TensorFlowServingModelConfig:
    """Parse TensorFlow Serving model configuration protobuf messages. Add
    or remove models from the message.
    """

    def __init__(self):
        self.config = None
        self._models = None

    def parse_config_file(self, filepath: str) -> dict:
        """Parses the protobuf message and returns it as a dictionary

        Returns:
            dict: Dict representation of the protobuf message.
        """     
        with open(filepath, "r") as file:
            config_file = file.read()
        message_format = model_server_config_pb2.ModelServerConfig()
        message = text_format.Parse(text=config_file, message=message_format)
        self.config: dict = json_format.MessageToDict(message)
        self._models: List[dict] = self.config["modelConfigList"]["config"]

    def initialise_blank_config(self):
        message = model_server_config_pb2.ModelServerConfig(
            model_config_list=model_server_config_pb2.ModelConfigList(
                config=[]
            )
        )
        self.config: dict = json_format.MessageToDict(message)
        self.config["modelConfigList"]["config"] = []
        self._models: List[dict] = self.config["modelConfigList"]["config"]

    @property
    def models(self) -> List[dict]:
        """Property accessor for the models in the model config protobuf

        Returns:
            dict: The list of models in the protobuf
        """
        return self._models

    def remove_model(self, model_name: str):
        if self.config is None:
            raise ModelConfigError("No model config file initialised.")
        model_names = [m["name"] for m in self._models]
        if model_name not in model_names:
            raise ModelConfigError("Model name does not exist in config file.")
        models = list(filter(lambda x: x["name"] != model_name, self._models))
        self._models = models

    def add_model(
        self,
        model_name: str,
        base_path: str,
        model_platform: str = "tensorflow",
    ):
        """Add a model to the config file.

        Args:
            model_name (str): Name of the model
            base_path (str): The base path of the model
            model_platform (str, optional): The model platform. Defaults to
                "tensorflow".

        Raises:
            ModelConfigError: Raised if the model name already exists in the    
                parsed config file.
            ModelConfigError: Raised if the model base path already exists in
                the parsed config file.
        """
        if self.config is None:
            raise ModelConfigError("No model config file initialised.")
        model_names = [m["name"] for m in self._models]
        base_paths = [m["basePath"] for m in self._models]
        if model_name in model_names:
            raise ModelConfigError("Model name already exists in config file.")
        if base_path in base_paths:
            raise ModelConfigError(
                "Model base path already exists in config file."
            )
        self._models.append(
            {
                "name": model_name,
                "basePath": base_path,
                "modelPlatform": model_platform,
            }
        )

    def to_proto(self) -> ModelServerConfig:
        """Returns the dict representation of the model config file as a
        protobuf message

        Returns:
            ModelServerConfig: Protobuf representation of the config file
        """
        return json_format.Parse(
            json.dumps(self.config),
            message=model_server_config_pb2.ModelServerConfig(),
            ignore_unknown_fields=False,
        )

    def save_protobuf(self, filepath: str):
        """Saves the parsed config file to a protobuf message

        Args:
            filepath (str): Filepath for the protobuf message
        """
        with open(filepath, "w") as file:
            file.write(str(self.to_proto()))
