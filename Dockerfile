FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel AS env

WORKDIR /matting

COPY requirements.txt .
RUN pip install --ignore-installed dvc
RUN pip install --ignore-installed dvc[gdrive]
RUN pip install --ignore-installed -r /matting/requirements.txt
RUN apt-get update && apt-get install libsndfile1-dev -y


FROM env
#COPY data .
COPY src ./src
#RUN git init
#RUN dvc init --no-scm
COPY .git ./.git
COPY .dvc ./.dvc
COPY datasets/*.dvc ./datasets/
RUN apt install git -y
RUN git init
RUN dvc pull
