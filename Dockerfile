FROM nvidia/cuda:11.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

ENV CUDNN_VERSION=8.0.5.39-1+cuda11.1
ENV NCCL_VERSION=2.7.8-1+cuda11.1

ARG python=3.8
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades \
    --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget \
    ca-certificates \
    libcudnn8=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libnccl-dev=${NCCL_VERSION} \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN /usr/bin/python -m pip install --upgrade pip

# Install pytorch
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 \
-f https://download.pytorch.org/whl/torch_stable.html 

RUN pip install tensorboard==2.5.0
RUN pip install tensorboard-data-server==0.6.1
RUN pip install tensorboard-plugin-wit==1.8.0
RUN pip install tensorboardX==1.8

RUN pip install timm==0.4.5
RUN pip install opencv-contrib-python-headless==4.5.2.54
RUN pip install tqdm==4.61.2
RUN pip install PyYAML==5.4.1
RUN pip install Pillow==8.3.1
RUN pip install einops==0.3.0
RUN pip install scipy==1.7.1




