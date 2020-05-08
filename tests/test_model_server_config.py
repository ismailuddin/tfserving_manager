from unittest import TestCase
import tempfile
from tensorflow_serving.config import model_server_config_pb2
from model_server.model_server_config import TensorFlowServingModelConfig
from model_server.exceptions import ModelConfigError


class TestTensorFlowServingModelConfig(TestCase):
    def setUp(cls):
        cls.models = [
            {"name": "model_A", "base_path": "/path/to/model/A/"},
            {"name": "B", "base_path": "/path/to/model/B/"},
        ]
        model_config = model_server_config_pb2.ModelServerConfig(
            model_config_list=model_server_config_pb2.ModelConfigList(
                config=[
                    model_server_config_pb2.ModelConfig(
                        name=m["name"], base_path=m["base_path"]
                    )
                    for m in cls.models
                ]
            )
        )
        cls.model_config_file = tempfile.NamedTemporaryFile()
        with open(cls.model_config_file.name, "w") as file:
            file.write(str(model_config))
        cls.t = TensorFlowServingModelConfig()

    def tearDown(cls):
        cls.model_config_file.close()

    def test_initialise_blank_config(self):
        self.t.initialise_blank_config()
        self.assertIsInstance(self.t.config, dict)
        self.assertIsInstance(self.t.models, list)

    def test_parse_config_file(self):
        self.t.parse_config_file(filepath=self.model_config_file.name)

    def test_models_property(self):
        self.t.parse_config_file(filepath=self.model_config_file.name)
        self.assertEqual(len(self.t.models), len(self.models))

    def test_add_model(self):
        self.t.initialise_blank_config()
        model_name, base_path = "model_C", "/path/tp/model/C/"
        self.t.add_model(model_name=model_name, base_path=base_path)
        models = self.t.models
        self.assertIn(model_name, [x["name"] for x in models])
        self.assertIn(base_path, [x["basePath"] for x in models])

    def test_add_model_exceptions(self):
        self.t.initialise_blank_config()
        model_name, base_path = "model_C", "/path/tp/model/C/"
        self.t.add_model(model_name=model_name, base_path=base_path)
        # Try to add model with same name
        with self.assertRaises(ModelConfigError):
            self.t.add_model(model_name=model_name, base_path=base_path)
        # Try to add model with same base path
        with self.assertRaises(ModelConfigError):
            self.t.add_model(model_name="model_D", base_path=base_path)

    def test_remove_model(self):
        self.t.parse_config_file(filepath=self.model_config_file.name)
        model_name, base_path = "model_C", "/path/tp/model/C/"
        self.t.add_model(model_name=model_name, base_path=base_path)
        self.t.remove_model(model_name="model_C")
        models = self.t.models
        self.assertNotIn(model_name, [x["name"] for x in models])

    def test_remove_model_exception(self):
        self.t.parse_config_file(filepath=self.model_config_file.name)
        # Try to remove non-existent model
        with self.assertRaises(ModelConfigError):
            self.t.remove_model(model_name="model_C")

    def test_to_proto(self):
        self.t.parse_config_file(filepath=self.model_config_file.name)
        protobuf = self.t.to_proto()
        self.assertIsInstance(
            protobuf, model_server_config_pb2.ModelServerConfig
        )

    def test_save_protobuf(self):
        self.t.parse_config_file(filepath=self.model_config_file.name)
        with tempfile.NamedTemporaryFile() as file:
            self.t.save_protobuf(file.name)
            # Test generated Protobuf is valid
            t = TensorFlowServingModelConfig()
            t.parse_config_file(filepath=file.name)
