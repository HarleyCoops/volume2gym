FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
# The public container exposes only pinned named volumes unless an operator
# explicitly opts into bounded custom-input compilation.
ENV SOURCE2AGENT_ALLOW_CUSTOM_INPUT=0
ENV SOURCE2AGENT_MAX_REQUEST_BYTES=262144
ENV SOURCE2AGENT_MAX_CUSTOM_UNITS=100

EXPOSE 8000

CMD ["python", "deploy/server.py"]
