import torch
from datasets import DatasetDict
from steps.configuration import HuggingfaceConfig
from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from zenml.steps import step


@step
def token_evaluator(
    config: HuggingfaceConfig,
    model: PreTrainedModel,
    tokenized_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
) -> float:
    """Evaluate trained model on validation set"""
    # Needs to recompile because we are reloading model for evaluation
    model.compile(optimizer=torch.nn.optim.Adam())

    validation_set = tokenized_datasets["validation"](
        columns=["attention_mask", "input_ids", "labels"],
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=DataCollatorForTokenClassification(
            tokenizer, return_tensors="pt"
        ),
    )

    # Calculate loss
    if config.dummy_run:
        test_loss = model.evaluate(validation_set.take(10), verbose=1)
    else:
        test_loss = model.evaluate(validation_set, verbose=1)
    return test_loss
