FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y libgl1-mesa-glx
#FROM ubuntu:18.04

#SHELL [ "/bin/bash", "--login", "-c" ]


WORKDIR /decimer-project
COPY requirements.txt /decimer-project

COPY decimer-conda-env.yml /decimer-project

RUN conda env create -f /decimer-project/decimer-conda-env.yml
RUN echo "source activate nenv" > ~/.bashrc
ENV PATH /opt/conda/envs/nenv/bin:$PATH

RUN pip install cython && pip install scikit-image && pip install django-filter && pip install django-storages
RUN pip install -r requirements.txt 
# --use-feature=2020-resolver
#RUN conda install --yes --file requirements.txt 

#RUN conda deactivate

COPY . /decimer-project

#RUN adduser -D user

EXPOSE 8000
ENTRYPOINT ["sh", "entrypoint.sh"]



