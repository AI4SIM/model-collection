FROM nvidia/cuda:11.3.1-cudnn8-devel-ubi8

LABEL maintainer="Atos AI4Sim"

RUN echo "alias ll='ls -la'" >> /root/.bashrc

ENV https_proxy=http://129.183.4.13:8080
ENV http_proxy=http://129.183.4.13:8080

RUN dnf update -y \
    && dnf install -y python39 \
                      git \
                      wget \
                      sudo \
    && dnf clean all \
    && rm -rf /var/cache/dnf/*

RUN ln -snf python3 /usr/bin/python
RUN ln -snf pip3 /usr/bin/pip

# RUN apt-get update -y \
# && apt-get install -y sudo zsh

ENV USER mluser
ENV HOME /home/${USER}
RUN adduser --create-home ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# RUN adduser --disabled-password --gecos "" ${USER} \
    # && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${USER}
WORKDIR ${HOME}
ENV PATH="${HOME}/.local/bin:${PATH}"
RUN echo "export PATH=${PATH}:${HOME}/.local/bin" >> ${HOME}/.bashrc

RUN pip install --upgrade pip
RUN pip install --user wheel
RUN pip install --user torch==1.10.2+cu113 \
                       torchvision==0.11.3+cu113 \
                       torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install --user torch-scatter \
                       torch-sparse \
                       torch-cluster \
                       torch-spline-conv \
                       torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN pip install --user h5py \
                       metaflow \
                       memory_profiler \
                       netCDF4 \
                       notebook \
                       nvitop \
                       pytorch-lightning \
                       PyYAML \
                       randomname \
                       torchmetrics \
                       torch_optimizer

RUN mkdir ${HOME}/.jupyter
COPY jupyter_server_config.py ${HOME}/.jupyter/

CMD ["bash"]