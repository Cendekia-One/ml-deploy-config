# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Install Git to clone the repository
RUN apt-get update \
    && apt-get install -y git

# Copy the local folder into the container
COPY . /app

# Install ODBC drivers for SQL Server
RUN apt-get install -y --no-install-recommends \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port if your application requires it
EXPOSE 80

# Run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
