FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY tgmanager ./tgmanager

VOLUME /data
ENTRYPOINT ["python", "-m", "tgmanager"]
