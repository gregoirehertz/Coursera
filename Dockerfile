# Use a more general Python base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Install build dependencies for Python packages (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy just the requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 80

# Define environment variable
ENV NAME Fraud

# Run app.py when the container launches
CMD ["python", "app.py"]
