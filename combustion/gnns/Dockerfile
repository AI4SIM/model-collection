##############################################################
# General variables used during all build stages
##############################################################

# Possibility to point to your own docker registry 
ARG DOCKER_REGISTRY=docker.io

##############################################################
# Final image
##############################################################

FROM ${DOCKER_REGISTRY}/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ARG USE_CASE_PATH=combustion/gnns
ARG PYTHON_VERS

# install required packages
RUN apt-get update \
 && apt-get install -y \
  python${PYTHON_VERS}-dev \
  python3-pip \
  build-essential \
  libssl-dev \
  libffi-dev \
  vim

RUN ln -s /usr/bin/python${PYTHON_VERS} /usr/bin/python
RUN pip install nox

# create user ai4sim
ARG uid=1000
ARG gid=1000
RUN groupadd -g ${gid} ai4sim \
  && useradd -u ${uid} -g ${gid} -m -s /bin/bash ai4sim

# Retrieve use-case related files
ARG AI4SIM_HOME=/home/ai4sim
ADD ${USE_CASE_PATH} ${AI4SIM_HOME}/${USE_CASE_PATH}
ADD tools ${AI4SIM_HOME}/tools
RUN chown -R ai4sim:ai4sim ${AI4SIM_HOME}

# change user
USER ai4sim
ENV PATH=${PATH}:/home/ai4sim/.local/bin

# Install all the python dependencies in the environment
WORKDIR ${AI4SIM_HOME}/${USE_CASE_PATH}
RUN python3 -m nox -s dev_dependencies --no-venv

CMD ["bash"]