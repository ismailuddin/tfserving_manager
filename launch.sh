#!/bin/bash
docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source="$(pwd)/models/build/",target=/models  tensorflow/serving --model_config_file=/models/model_config.proto
