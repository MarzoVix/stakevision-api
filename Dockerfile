FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install headless opencv FIRST, then paddleocr with --no-deps to prevent
# it from pulling in opencv-python/opencv-contrib-python
RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84 && \
    pip install --no-cache-dir paddlepaddle==2.6.2 && \
    pip install --no-cache-dir --no-deps paddleocr==2.9.1 && \
    pip install --no-cache-dir pyclipper shapely scikit-image imgaug lmdb lxml beautifulsoup4 rapidfuzz python-docx && \
    pip install --no-cache-dir fastapi uvicorn python-multipart Pillow

COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
