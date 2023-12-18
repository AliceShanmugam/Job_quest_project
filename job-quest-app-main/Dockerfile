FROM python:3.10.6-slim-buster

WORKDIR /prod

COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

COPY scripts/params.py scripts/params.py
COPY scripts/model/registry.py scripts/model/registry.py
COPY scripts/preprocessor/preprocessing_query.py scripts/preprocessor/preprocessing_query.py
COPY api api
COPY outputs outputs

CMD uvicorn api.main:app --host 0.0.0.0
