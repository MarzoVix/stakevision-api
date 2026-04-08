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
COPY constraints.txt .
COPY requirements.txt .

# Use -c constraints.txt on EVERY pip install to force numpy==1.26.4
# This cannot be overridden by any dependency
ENV PIP_CONSTRAINT=/app/constraints.txt

RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84
RUN pip install --no-cache-dir paddlepaddle==2.6.2
RUN pip install --no-cache-dir --no-deps paddleocr==2.9.1
RUN pip install --no-cache-dir \
    imgaug albumentations requests pyclipper shapely scikit-image \
    lmdb lxml beautifulsoup4 rapidfuzz python-docx pyyaml tqdm \
    fire cython openpyxl attrdict3 premailer
RUN pip install --no-cache-dir fastapi uvicorn python-multipart Pillow

COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
