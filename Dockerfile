# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Chainlit default port
EXPOSE 8000

# Run the app
CMD ["chainlit", "run", "app.py", "--port", "8000", "--host", "0.0.0.0"]
