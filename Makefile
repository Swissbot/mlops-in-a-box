.PHONY: setup train serve test mlflow-ui clean

setup:
	python3.12 -m venv .venv
	. .venv/bin/activate && python -m pip install -U pip setuptools wheel
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -e .

train:
	. .venv/bin/activate && mlbox-train

serve:
	. .venv/bin/activate && uvicorn mlops_box.serving.app:app --host 0.0.0.0 --port 8000

test:
	. .venv/bin/activate && pytest -q

mlflow-ui:
	. .venv/bin/activate && mlflow ui --host 0.0.0.0 --port 5000

clean:
	rm -rf .venv runs models mlruns __pycache__ .pytest_cache
