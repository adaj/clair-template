# Base.Dockerfile

# Use a pinned version of the base image for deterministic behavior
FROM tiangolo/uvicorn-gunicorn:python3.9-slim AS base

LABEL author="Adelson de Araujo (a.dearaujo@utwente.nl)"

WORKDIR /clair

# First, copy only requirements.txt and install dependencies
COPY ./clair/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get update \
    && apt-get install -y supervisor
