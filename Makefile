.PHONY: install train test run api figures
install:
	pip install -r requirements.txt
train:
	python train_models.py
test:
	python -m unittest discover -p "test_*.py" -v
run:
	streamlit run dashboard.py --browser.gatherUsageStats=false
api: train
	uvicorn api:app --host 127.0.0.1 --port 8000
figures:
	python make_figures.py
