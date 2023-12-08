# Stage 1: Build
FROM python:3.8-alpine as builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache build-base

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase
COPY . .

# Stage 2: Final Image
FROM python:3.8-alpine

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /app /app

# Set non-root user
RUN adduser -D appuser
USER appuser

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=80

# Expose port for the Flask app
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "python", "-c", "import requests; requests.get('http://localhost')" ]

# Run the Flask app
CMD ["flask", "run"]
