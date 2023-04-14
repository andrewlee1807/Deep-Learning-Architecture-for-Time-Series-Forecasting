#FROM python:3.8-slim
#
## set the working directory in the container
#WORKDIR /app
#
## copy the dependencies file to the working directory
#COPY ./requirements.txt /app
#RUN pip install --no-cache-dir -r requirements.txt
#
## copy the content of the local src directory to the working directory
#COPY . /app
#
## command to run on container start
#CMD [ "python", "main.py", "--dataset_name=spain", "--model_name=Model1", "--config_path=config/spain_model_1.yaml", "--output_length=1", "--device=0", "--output_dir=results/spain_vanilla_model_1"]


FROM continuumio/miniconda3


RUN conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

RUN python -m pip install "tensorflow<2.11"

WORKDIR /app

SHELL ["/bin/bash", "--login", "-c"]


## Create the environment:
#COPY ./_env.yml /app
#RUN conda env create -f _env.yml
COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
#
## Initialize conda in bash config fiiles:
#RUN conda init bash
#
## Activate the environment, and make sure it's activated:
#RUN conda activate ts_model_test
RUN echo "Make sure tensorflow is installed:"
RUN python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"

RUN sleep 10

# copy the content of the local src directory to the working directory
COPY . /app

# command to run on container start
CMD [ "python", "main.py", "--dataset_name=spain", "--model_name=Model1", "--config_path=config/spain_model_1.yaml", "--output_length=1", "--device=0", "--output_dir=results/spain_vanilla_model_1"]

