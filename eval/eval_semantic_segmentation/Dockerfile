ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all
RUN pip install --no-cache-dir --upgrade pip wheel setuptools


RUN git clone https://github.com/NVIDIA/apex /apex
WORKDIR /apex
ENV FORCE_CUDA="1"
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


RUN pip install -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html mmcv-full==1.2.7
RUN pip install --no-cache-dir pytest-runner scipy tensorboardX faiss-gpu==1.6.1 tqdm lmdb sklearn pyarrow==2.0.0 timm DALL-E munkres six einops


RUN git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection /Swin-Transformer-Object-Detection
WORKDIR /Swin-Transformer-Object-Detection
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -v -e .


RUN git clone -b v0.11.0 https://github.com/open-mmlab/mmsegmentation /mmsegmentation
WORKDIR /mmsegmentation
RUN pip install --no-cache-dir -v -e .
WORKDIR /workspace