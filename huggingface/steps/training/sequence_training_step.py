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
from datasets import DatasetDict
from sklearn.metrics import accuracy_score
from steps.configuration import HuggingfaceConfig
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
    Trainer,
    PreTrainedModel,
    TrainingArguments,
    IntervalStrategy,
    EarlyStoppingCallback
)

from zenml.steps import step


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

@step
def sequence_trainer(
    config: HuggingfaceConfig,
    tokenized_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedModel:
    """Build and Train token classification model"""

    # Get label list
    label_list = tokenized_datasets["train"].unique("label")

    # Load pre-trained model from huggingface hub
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model, num_labels=len(label_list)
    )

    num_train_steps = (
        len(tokenized_datasets["train"]) // config.batch_size
    ) * config.epochs


    args = TrainingArguments(
        output_dir = "/tmp/sequence_classification/",
        per_device_train_batch_size = config.batch_size,
        per_device_eval_batch_size = config.batch_size,
        learning_rate = config.init_lr,
        evaluation_strategy = IntervalStrategy.STEPS,
        eval_steps = 50,
        save_total_limit = 5,
        weight_decay = 0.1,
        adam_epsilon = 1e-8,
        max_grad_norm = 1.0,
        num_train_epochs = 100.0,
        warmup_steps = num_train_steps * 0.1,
        logging_steps = 1000,
        save_steps = 3500,
        no_cuda = True,
        seed = 42,
        local_rank = -1,
        metric_for_best_model = 'accuracy',
        load_best_model_at_end=True
    )

    columns_to_remove = [
        feature for feature
        in tokenized_datasets["train"].features
        if feature not in ["input_ids", "attention_mask", "label"]
    ]

    train_set = tokenized_datasets["train"].remove_columns(columns_to_remove)
    test_set = tokenized_datasets["test"].remove_columns(columns_to_remove)

    trainer = Trainer(
        data_collator = DataCollatorWithPadding(tokenizer = tokenizer),
        compute_metrics = compute_metrics,
        model = model,
        args = args,
        train_dataset = train_set,
        eval_dataset = test_set,
        tokenizer = tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold = 0.05)]
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    if config.dummy_run:
        trainer.train_dataset = train_set.select(list(range(0,10)))
        trainer.args.num_train_epochs = 1
    else:
        trainer.train_dataset = train_set
        trainer.args.num_train_epochs = config.epochs

    trainer.train()
    return trainer.model
