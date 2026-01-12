FROM python:3.12-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY pyproject.toml .
COPY src/ ./src

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-cache-dir

RUN mkdir -p /models /reports/figures

ENTRYPOINT ["python", "-u", "-m", "my_project.evaluate"]
