##############################################################
# General variables used during all build stages
##############################################################

# Possibility to point to your own docker registry 
ARG DOCKER_REGISTRY=docker.io
ARG UBUNTU_IMAGE_NAME=nvidia/cuda
ARG UBUNTU_IMAGE_TAG=12.8.1-cudnn-runtime-ubuntu22.04

##############################################################
# Final image
##############################################################

FROM ${DOCKER_REGISTRY}/${UBUNTU_IMAGE_NAME}:${UBUNTU_IMAGE_TAG}

ARG MODEL_PROJECT_PATH
ARG PYTHON_VERSION

# install required packages
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update \
 && apt-get install -y \
  software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y \
  python${PYTHON_VERSION} \
  python3-pip \
  build-essential \
  libssl-dev \
  libffi-dev \
  vim \
  git \
 && apt-get clean

# re-link python3 to the proper python version
RUN rm -rf /usr/bin/python3 \
  && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# create user ai4sim
ARG uid=1000
ARG gid=1000
ARG uname=ai4sim
ARG gname=ai4sim
ARG AI4SIM_HOME=/home/${uname}
RUN groupadd -g ${gid} ${gname} \
  && useradd -u ${uid} -g ${gid} -m -s /bin/bash ${uname} \
  && chown -R ${uname}:${gname} ${AI4SIM_HOME}

# Retrieve use-case related files
ADD --chown=${uname}:${gname} ${MODEL_PROJECT_PATH} ${AI4SIM_HOME}/${MODEL_PROJECT_PATH}
ADD --chown=${uname}:${gname} tools ${AI4SIM_HOME}/tools

# change user
USER ${uname}
ENV PATH=${PATH}:${AI4SIM_HOME}/.local/bin

# Install all the python dependencies in the environment
WORKDIR ${AI4SIM_HOME}/${MODEL_PROJECT_PATH}
ENV UV_NO_CACHE=1
RUN python3 -m pip install nox$(grep 'nox.needs_version' ./noxfile.py | sed -rn 's/.*\"([><=]=.*)\"/\1/p') \
 && python3 -m pip install uv \
 && python3 -m pip cache purge \
 && python3 -m nox -s dev_dependencies --no-venv

CMD ["bash"]