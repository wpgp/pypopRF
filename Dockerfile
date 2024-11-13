# Use Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY . .

# Install package
RUN pip install --no-cache-dir -e ".[dev]"

# Create directories for data and output
RUN mkdir -p data output

# Set default command
ENTRYPOINT ["pypoprf"]
CMD ["--help"]