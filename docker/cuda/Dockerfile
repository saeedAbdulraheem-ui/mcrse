FROM nvidia/cuda:12.8.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.9.0

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
	python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        libcudnn8 \
        libcudnn8-dev \
        pkg-config \
        ## Python
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    wget build-essential libssl-dev zlib1g-dev \
    libncurses5-dev libncursesw5-dev libreadline-dev \
    libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev \
    libexpat1-dev liblzma-dev tk-dev libffi-dev \
    && cd /usr/src \
    && wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar xzf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && python3.10 -m ensurepip \
    && python3.10 -m pip install --upgrade pip \
    && rm -rf /usr/src/Python-3.10.14*

# Optional: use python3.10 and pip3.10 as system defaults
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.10 1

# RUN cd /opt/ &&\
#     # Download and unzip OpenCV and opencv_contrib and delte zip files
#     wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
#     unzip $OPENCV_VERSION.zip &&\
#     rm $OPENCV_VERSION.zip &&\
#     wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
#     unzip ${OPENCV_VERSION}.zip &&\
#     rm ${OPENCV_VERSION}.zip &&\
#     # Create build folder and switch to it
#     mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
#     # Cmake configure
#     cmake \
#         -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
#         -DWITH_CUDA=ON \
#         -DWITH_CUDNN=ON \
#         -DOPENCV_DNN_CUDA=ON \
#         -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
#         -DCMAKE_BUILD_TYPE=RELEASE \
#         # Install path will be /usr/local/lib (lib is implicit)
#         -DCMAKE_INSTALL_PREFIX=/usr/local \
#         .. &&\
#     # Make
#     make -j"$(nproc)" && \
#     # Install to /usr/local/lib
#     make install && \
#     ldconfig &&\
#     # Remove OpenCV sources and build folder
#     rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}


WORKDIR /storage
COPY ../requirements.txt /storage/requirements.txt
RUN python3.10 -m pip install -r requirements.txt

RUN apt update && apt-get install -y libcublas-12-0