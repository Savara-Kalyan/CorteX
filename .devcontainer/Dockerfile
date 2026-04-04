
FROM mcr.microsoft.com/devcontainers/python:3.12

ENV PYTHONUNBUFFERED=1


RUN apt-get update && \
    apt-get install -y curl libmagic-dev poppler-utils

RUN pip install --upgrade pip

