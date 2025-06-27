FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    build-essential \
    pkg-config \
    git \
    libfreetype6-dev \
    libpng-dev \
    libqhull-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py \
    && python3.7 get-pip.py \
    && rm get-pip.py

RUN ln -sf /usr/bin/python3.7 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.7 /usr/bin/pip \
    && ln -sf /usr/local/bin/pip3.7 /usr/bin/pip3

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir numpy==1.21.6

RUN pip install --no-cache-dir \
    matplotlib==2.1.1 \
    scikit-learn==1.0.2 \
    seaborn==0.9.0

RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir \
    torch-scatter==2.0.2 \
    torch-sparse==0.6.1 \
    torch-cluster==1.5.4 \
    torch-spline-conv==1.2.0 \
    torch-geometric==1.4.3 \
    --find-links https://data.pyg.org/whl/torch-1.4.0+cpu.html

RUN pip install --no-cache-dir \
    deepsnap==0.1.2 \
    networkx==2.4 \
    test-tube==0.7.5 \
    tqdm==4.43.0

COPY . .
