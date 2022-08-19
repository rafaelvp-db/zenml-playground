.PHONY: env init run install mlflow stack clean

clean:
	zenml stack down && \
	zenml stack set default && \
	zenml stack delete mlflow_stack -y && \
	zenml experiment-tracker delete mlflow_tracker && \
	zenml artifact-store delete local_artifact_store && \
	zenml metadata-store delete sqlite_meta_store && \
	rm -rf .data

mlflow:
	zenml experiment-tracker register mlflow_tracker --flavor=mlflow && \
	zenml artifact-store register local_artifact_store --flavor=local --path=/Users/rafael.pierre/projects/zenml-playground/huggingface/.data/mlruns && \
	zenml metadata-store register sqlite_meta_store --flavor=sqlite --uri=/Users/rafael.pierre/projects/zenml-playground/huggingface/.data/mlflow.sqlite && \
	zenml stack register mlflow_stack \
    -m sqlite_meta_store \
    -a local_artifact_store \
    -o default \
    -e mlflow_tracker --set

env:
	python -m venv .venv && \
	source ../.venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r ../requirements.txt

install:
	zenml integration install pytorch huggingface mlflow

init: env install
	source ../.venv/bin/activate && \
	zenml init

run:
	source ../.venv/bin/activate && \
	zenml pipeline run pipelines/sequence_classifier_pipeline/sequence_classifier_pipeline.py -c sequence_classification_config.yaml