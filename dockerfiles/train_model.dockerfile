# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY leaf_shapes/ ./leaf_shapes/
COPY Makefile Makefile
COPY config.yaml config.yaml
WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_dev.txt --no-cache-dir

RUN pip install . --no-deps --no-cache-dir

# Test to fix "no module found error for models/model.py script"
ENV PYTHONPATH /

# For dvc data
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY data/ data/
RUN dvc config core.no_scm true

ENTRYPOINT ["make", "train"]
