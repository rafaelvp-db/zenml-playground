#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from zenml.steps.base_step_config import BaseStepConfig
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

import mlflow

class HuggingfaceConfig(BaseStepConfig):
    """Config for the token-classification"""

    pretrained_model = "distilbert-base-uncased"
    batch_size = 16
    epochs = 1
    dummy_run = True
    init_lr = 2e-5
    weight_decay_rate = 0.01
    text_column = "tokens"  # "text" for Sequence
    label_column = "ner_tags"  # "label" for Sequence
    label_all_tokens = True  # Irrelevant for Sequence
    dataset_name = "conll2003"  # "imdb" for Sequence
    max_seq_length = 128  # Irrelevant for Token Classification


class MLflowConfig():
    """Config for MLflow"""
    def __init__(
        self,
        experiment_name: str
    ):
        mlflow.set_tracking_uri(get_tracking_uri())
        self.experiment_name = experiment_name
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        self.experiment_id = None
        if not self.experiment:
            self.experiment_id = mlflow.create_experiment(name = experiment_name)
            self.experiment = mlflow.get_experiment(experiment_id = self.experiment_id)
        self.experiment = mlflow.set_experiment(experiment_id = self.experiment.experiment_id)
        

