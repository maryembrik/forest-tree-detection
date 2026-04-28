PYTHON=python
PIP=pip

.PHONY: setup train predict evaluate app test lint docker-build docker-run

setup:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) src/train.py --config config.yaml

predict:
	$(PYTHON) src/inference.py --config config.yaml --image $(IMAGE)

evaluate:
	$(PYTHON) src/metrics.py --config config.yaml

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v

lint:
	flake8 src/ app/ tests/ --max-line-length=100

docker-build:
	docker build -t forest-tree-detection .

docker-run:
	docker-compose up

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
