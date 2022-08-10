.PHONY: env init run activate install mlflow stack clean

activate:
	source .venv/bin/activate

clean:
	zenml stack down && \
	zenml stack set default && \
	zenml stack delete mlflow_stack -y && \
	zenml experiment-tracker delete mlflow_tracker && \
	rm -rf mlruns

stack:
	zenml stack register mlflow_stack \
    -m sqlitedb \
    -a default \
    -o default \
    -e mlflow_tracker --set

mlflow:
	zenml experiment-tracker register mlflow_tracker --flavor=mlflow && \
	zenml stack register mlflow_stack \
    -m default \
    -a default \
    -o default \
    -e mlflow_tracker --set

env:
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

install:
	zenml integration install pytorch huggingface mlflow

init: env install
	source .venv/bin/activate && \
	zenml init

run:
	source .venv/bin/activate && \
	mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root ./mlruns & \
	zenml pipeline run pipelines/sequence_classifier_pipeline/sequence_classifier_pipeline.py -c sequence_classification_config.yaml