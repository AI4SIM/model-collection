##############################################################
# Temporary image to download necessary files
##############################################################

FROM ubuntu AS downloader

# install required packages
RUN apt-get update \
 && apt-get install -y git

RUN mkdir -p /tmp/ai4sim/
WORKDIR /tmp/ai4sim/

# clone ai4sim code
RUN git clone https://github.com/AI4SIM/model-collection.git

##############################################################
# Final image
##############################################################

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# install required packages
RUN apt-get update \
 && apt-get install -y git \
 python3.8-dev \
 python3-pip \
 build-essential \
 libssl-dev \
 libffi-dev


RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN pip install nox

# create user ai4sim
ARG uid=1000
ARG gid=1000
RUN groupadd -g ${gid} ai4sim \
  && useradd -u ${uid} -g ${gid} -m -s /bin/bash ai4sim

WORKDIR /home/ai4sim

ARG USE_CASE_PATH=model-collection/combustion/unets

RUN mkdir -p $USE_CASE_PATH

COPY --from=downloader /tmp/ai4sim/$USE_CASE_PATH $USE_CASE_PATH
COPY --from=downloader /tmp/ai4sim/model-collection/tools/  ./model-collection/tools/

WORKDIR $USE_CASE_PATH
RUN python3 -m nox -s dev_dependencies --no-venv

# change user
USER ai4sim

CMD ["bash"]