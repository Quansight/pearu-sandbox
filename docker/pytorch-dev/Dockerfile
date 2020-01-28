#
# Usage:
#  docker build -t pytorch-dev:v2 .
#  docker run --gpus all pytorch-dev:v2 bash -l test.sh
#

FROM nvidia/cuda:9.2-devel-centos7
ENV CUDA_HOME=/usr/local/cuda/

RUN yum update -y && yum install -y \
        sudo \
        curl \
        git \
        bc \
        which

RUN adduser --shell /bin/bash user \
        && echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user \
        && chmod 777 /home/user

USER user
ENV HOME=/home/user
WORKDIR /home/user

RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        && chmod +x ~/miniconda.sh \
        && ~/miniconda.sh -b -p ~/miniconda \
        && rm ~/miniconda.sh \
        && ~/miniconda/bin/conda init bash

ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

COPY pytorch-cuda92-dev.yaml .

ENV CONDA_ENV=pytorch-cuda92-dev
RUN conda env create --file=pytorch-cuda92-dev.yaml -n $CONDA_ENV && conda clean -ya

RUN git clone https://github.com/pytorch/pytorch.git \
        && cd pytorch \
        && git submodule sync --recursive \
        && git submodule update --init --recursive

RUN sudo yum install -y which

COPY build.sh .
RUN /bin/bash -l build.sh

COPY test_issue30488.sh test.sh
#RUN /bin/bash -l test.sh

