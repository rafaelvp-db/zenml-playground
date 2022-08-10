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
import torch
from torch import nn
from datasets import DatasetDict
from steps.configuration import HuggingfaceConfig
from transformers import (
    pipeline,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    PreTrainedModel
)

import evaluate
from evaluate import evaluator

from zenml.steps import step


@step
def sequence_evaluator(
    config: HuggingfaceConfig,
    model: PreTrainedModel,
    tokenized_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
) -> float:
    """Evaluate trained model on validation set"""

    columns_to_remove = [
        feature for feature
        in tokenized_datasets["test"].features
        if feature not in ["text", "label"]
    ]

    validation_set = tokenized_datasets["test"] \
        .remove_columns(columns_to_remove)

    dummy_validation_set = validation_set \
        .select(list(range(0,10)))

    print(columns_to_remove)

    # Calculate loss

    task_evaluator = evaluator("text-classification")
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

    if config.dummy_run:
        metrics = task_evaluator.compute(
            tokenizer = tokenizer,
            model_or_pipeline = pipe,
            data = dummy_validation_set,
            label_mapping={"LABEL_0": 0, "LABEL_1": 1},
            metric = evaluate.combine(["accuracy", "recall", "precision", "f1"]),
        )
    else:
        metrics = task_evaluator.compute(
            tokenizer = tokenizer,
            model_or_pipeline = pipe,
            data = dummy_validation_set,
            label_mapping={"LABEL_0": 0, "LABEL_1": 1},
            metric = evaluate.combine(["accuracy", "recall", "precision", "f1"]),
        )
    return metrics["accuracy"]
