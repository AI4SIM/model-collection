##############################################################
# General variables used during all build stages
##############################################################

# Possibility to point to your own docker registry 
ARG DOCKER_REGISTRY=docker.io
ARG UBUNTU_IMAGE_NAME
ARG UBUNTU_IMAGE_TAG

##############################################################
# Final image
##############################################################

FROM ${DOCKER_REGISTRY}/${UBUNTU_IMAGE_NAME}:${UBUNTU_IMAGE_TAG}

ARG USE_CASE_PATH
ARG PYTHON_VERS

# install required packages
RUN apt-get update \
 && apt-get install -y \
  python${PYTHON_VERS}-dev \
  python3-pip \
  build-essential \
  libssl-dev \
  libffi-dev \
  vim \
 && apt-get clean

# create user ai4sim
ARG uid=1000
ARG gid=1000
ARG AI4SIM_HOME=/home/ai4sim
RUN groupadd -g ${gid} ai4sim \
  && useradd -u ${uid} -g ${gid} -m -s /bin/bash ai4sim \
  && chown -R ai4sim:ai4sim ${AI4SIM_HOME}

# Retrieve use-case related files
ADD --chown=ai4sim:ai4sim ${USE_CASE_PATH} ${AI4SIM_HOME}/${USE_CASE_PATH}
ADD --chown=ai4sim:ai4sim tools ${AI4SIM_HOME}/tools

# change user
USER ai4sim
ENV PATH=${PATH}:/home/ai4sim/.local/bin

# Install all the python dependencies in the environment
WORKDIR ${AI4SIM_HOME}/${USE_CASE_PATH}
RUN pip install nox \
 && python3 -m nox -s dev_dependencies --no-venv -- force_GPU_support purge

CMD ["bash"]