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

# Pin numpy 1.x FIRST - imgaug uses np.sctypes which was removed in numpy 2.0
# No version of imgaug on PyPI fixes this, so numpy MUST stay <2
RUN pip install --no-cache-dir "numpy==1.26.4"

# Headless opencv (no libGL needed)
RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84

# PaddlePaddle engine
RUN pip install --no-cache-dir paddlepaddle==2.6.2

# PaddleOCR without deps (prevents opencv-python override)
# Then install its deps manually, with numpy already locked to 1.26.4
RUN pip install --no-cache-dir --no-deps paddleocr==2.9.1 && \
    pip install --no-cache-dir \
    imgaug requests pyclipper shapely scikit-image lmdb lxml \
    beautifulsoup4 rapidfuzz python-docx pyyaml tqdm fire cython

# App deps
RUN pip install --no-cache-dir fastapi uvicorn python-multipart Pillow

COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
