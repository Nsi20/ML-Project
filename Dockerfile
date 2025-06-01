FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_RETRIES=10 \
    PIP_TIMEOUT=100

WORKDIR /app

# Install system dependencies and configure pip
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install requirements (ignore hashes)
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps --ignore-installed -r requirements.txt

# Copy application code
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]