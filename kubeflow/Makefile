.PHONY: env stack kubeflow run

env:
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install zenml notebook && \
	zenml integration install kubeflow tensorflow

stack:
	zenml container-registry register local_registry --flavor=default --uri=localhost:5000 && \
	zenml orchestrator register local_kubeflow_orchestrator --flavor=kubeflow && \
	zenml stack register local_kubeflow_stack \
		-m default \
		-a default \
		-o local_kubeflow_orchestrator \
		-c local_registry \
		--set

kubeflow:
	zenml stack up

stop:
	kubectl delete ns kubeflow

run:
	python run.py