# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_dev.txt --no-cache-dir

COPY pyproject.toml pyproject.toml
COPY leaf_shapes/ leaf_shapes/
COPY Makefile Makefile
COPY config.yaml config.yaml
RUN pip install . --no-deps --no-cache-dir

ENV PYTHONPATH /

# For dvc data
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY data/ data/
COPY models/ models/
RUN dvc config core.no_scm true

EXPOSE $PORT
ENTRYPOINT ["make", "run_api"]
