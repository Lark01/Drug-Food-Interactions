# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 7860

# Command to run the app
CMD ["streamlit", "run", "deploy.py", "--server.port=7860", "--server.address=0.0.0.0"]