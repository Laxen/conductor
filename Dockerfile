FROM python:3.11-slim

ARG BUILD_VERSION

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

LABEL \
    io.hass.version="${BUILD_VERSION}" \
    io.hass.type="app" \
    io.hass.arch="aarch64|amd64|armv7|i386"

CMD ["python", "main.py"]