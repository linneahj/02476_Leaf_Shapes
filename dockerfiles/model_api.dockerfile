# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY pyproject.toml pyproject.toml
COPY leaf_shapes/ leaf_shapes/
RUN pip install . --no-deps --no-cache-dir

COPY models/model_best.pth.tar model_best.pth.tar
COPY data/processed/Class_ids.csv Class_ids.csv

ENV MODEL_PATH=/model_best.pth.tar
ENV CLASS_IDS_PATH=/Class_ids.csv

EXPOSE $PORT
CMD exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker leaf_shapes.main:app
