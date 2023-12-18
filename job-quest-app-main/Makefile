# ----------------------------------
#         LOCAL SET UP
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

streamlit:
	-@streamlit run app/🏠_Home.py

run_api:
	uvicorn api.main:app --reload

train_ml:
	python scripts/main_ml.py

train_deep_dense:
	python scripts/main.py dense

train_deep_conv:
	python scripts/main.py conv
