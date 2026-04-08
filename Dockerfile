FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Headless opencv (no libGL needed)
RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84

# 2) PaddlePaddle engine
RUN pip install --no-cache-dir paddlepaddle==2.6.2

# 3) PaddleOCR without deps (prevents opencv-python override)
#    Then install its actual deps manually
#    imgaug>=0.4.0 fixes the np.sctypes AttributeError introduced in NumPy 2.0
RUN pip install --no-cache-dir --no-deps paddleocr==2.9.1 && \
    pip install --no-cache-dir \
    "numpy>=1.24" "imgaug>=0.4.0" \
    requests pyclipper shapely scikit-image lmdb lxml \
    beautifulsoup4 rapidfuzz python-docx pyyaml tqdm fire cython

# 4) App deps
RUN pip install --no-cache-dir fastapi uvicorn python-multipart Pillow

COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
