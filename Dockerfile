FROM python:3.9

# install required packages
RUN apt-get update \
 && apt-get install -y git

# create user ai4sim
ARG uid=1000
ARG gid=1000
RUN groupadd -g ${gid} ai4sim \
  && useradd -u ${uid} -g ${gid} -m -s /bin/bash ai4sim

# create workdir
USER ai4sim
WORKDIR /home/ai4sim

# clone ai4sim code
RUN git clone https://github.com/AI4SIM/model-collection.git

# install requirements
RUN python -m pip install -r model-collection/requirements.txt \
    -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
    --extra-index-url https://download.pytorch.org/whl/cu113

CMD ['bash']