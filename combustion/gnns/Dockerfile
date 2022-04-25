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

# change user
USER ai4sim
WORKDIR /home/ai4sim

# clone ai4sim code
RUN git clone https://github.com/AI4SIM/model-collection.git

ARG USE_CASE_PATH=model-collection/weather_forecast
WORKDIR $USE_CASE_PATH

RUN python3 -m nox -s dev_dependencies --no-venv

CMD ["bash"]
