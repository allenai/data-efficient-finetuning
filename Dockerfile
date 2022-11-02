FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# for jsonnet
RUN apt-get -y update
RUN apt-get install build-essential -y

WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt

# only copy the relevant folders
COPY attribution attribution
COPY scripts scripts
COPY training_config training_config
COPY shell_scripts shell_scripts
COPY qasper_train_and_eval.sh qasper_train_and_eval.sh