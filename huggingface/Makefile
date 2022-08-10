.PHONY: env init run activate

activate:
	source .venv/bin/activate

env:
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

init: env
	source .venv/bin/activate && \
	zenml integration install pytorch huggingface && \
	zenml init

run:
	source .venv/bin/activate && \
	zenml pipeline run pipelines/sequence_classifier_pipeline/sequence_classifier_pipeline.py -c sequence_classification_config.yaml