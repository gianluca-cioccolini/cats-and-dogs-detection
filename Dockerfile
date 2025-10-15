FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    unzip \
    ffmpeg \
    build-essential \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip3 install ultralytics 
RUN pip3 install roboflow streamlit opencv-python
RUN pip3 install matplotlib numpy
RUN pip3 install pandas

ENV YOLO_CONFIG_DIR=/workspace/.ultralytics
RUN mkdir -p /workspace/.ultralytics

EXPOSE 8501