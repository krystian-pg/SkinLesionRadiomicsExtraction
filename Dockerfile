FROM python:3.7

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Creating main directory
WORKDIR /app

# Copying packages to install 
COPY requirements.txt ./

# Installing all necessary packages
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Copying current directory to main container directory
COPY . .
