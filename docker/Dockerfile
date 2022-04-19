FROM python:3.9

# install required packages
RUN apt-get update \
 && apt-get install -y git

RUN pip install nox

# create user ai4sim
ARG uid=1000
ARG gid=1000
RUN groupadd -g ${gid} ai4sim \
  && useradd -u ${uid} -g ${gid} -m -s /bin/bash ai4sim

# copy script file
COPY ./script.sh /home/ai4sim
RUN chmod 755 /home/ai4sim/script.sh

# change user
USER ai4sim
WORKDIR /home/ai4sim

# clone ai4sim code
RUN git clone https://github.com/AI4SIM/model-collection.git

ENTRYPOINT ["./script.sh"]