# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone YOLOv7 repository
RUN git clone https://github.com/WongKinYiu/yolov7.git

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory into the container at /app
COPY . .

# Navigate to the YOLOv7 directory
WORKDIR /app/yolov7

# Install YOLOv7 dependencies
RUN pip install -r requirements.txt
RUN pip install --upgrade pip 
# Expose the port for Streamlit

# Run the Streamlit application
CMD ["streamlit", "run", "../app.py"]