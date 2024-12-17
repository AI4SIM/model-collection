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

ARG MODEL_PROJECT_PATH
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
RUN ln -sf /usr/bin/python${PYTHON_VERS} /usr/bin/python3

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
RUN python3 -m pip install $(grep nox== ${AI4SIM_HOME}/${MODEL_PROJECT_PATH}/requirements.txt || echo "nox") \
 && python3 -m nox -s dev_dependencies --no-venv -- purge

CMD ["bash"]