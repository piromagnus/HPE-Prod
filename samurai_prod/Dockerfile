FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# Set the working directory

# Set environment variables
# ENV ENV_NAME=sapiens
# ENV PYTHON_VERSION=3.10

# Update and install basic utilities
RUN apt-get update && apt-get install -y \
    fish \
    git \
    ninja-build \
    build-essential \
    gcc \
    g++ \
    ninja-build \
    curl \
    wget \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1-mesa-glx\
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# RUN pip install chumpy scipy munkres tqdm cython fsspec yapf==0.40.1 matplotlib packaging omegaconf ipdb ftfy regex
# RUN pip install json_tricks terminaltables modelindex prettytable albumentations opencv-python pandas

RUN git clone https://github.com/yangchris11/samurai.git

WORKDIR /samurai/sam2

RUN pip install -e . 
#&& pip install -e ".[notebooks]"
COPY requirements.txt /samurai/sam2/requirements.txt
RUN pip install -r /samurai/sam2/requirements.txt

RUN cd checkpoints && ./download_ckpts.sh && cd ..

WORKDIR /samurai

COPY track.py /samurai/track.py

RUN mkdir -p /data/videos

# Define mount points

# Set the entrypoint to run the script
ENTRYPOINT ["python3","track.py","--video_path","/data/videos"]