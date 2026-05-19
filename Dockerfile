FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling 2GB+ of CUDA libs)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    ultralytics==8.3.* \
    opencv-python-headless \
    paho-mqtt \
    Pillow

# Download YOLOv8n model at build time
RUN mkdir -p /models && \
    python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')" && \
    find / -name "yolov8n.pt" -exec cp {} /models/yolov8n.pt \; && \
    ls -la /models/yolov8n.pt

COPY cat_detector.py /app/cat_detector.py
COPY train.py /app/train.py

WORKDIR /app

CMD ["python3", "cat_detector.py"]
