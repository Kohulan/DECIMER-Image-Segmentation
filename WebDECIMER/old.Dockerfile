FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y libgl1-mesa-glx gcc python3.7


WORKDIR /WebDECIMER
COPY requirements.txt /WebDECIMER

COPY decimer-conda-env.yml /WebDECIMER



RUN conda env create -f /WebDECIMER/decimer-conda-env.yml
RUN echo "source activate DECIMER_IMGSEG" > ~/.bashrc
ENV PATH /opt/conda/envs/nenv/bin:$PATH


# decimer-backend-docker --python=python3.7
#ENV VIRTUAL_ENV=/WebDECIMER/decimer-venv-docker
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"


RUN apt-get install -y build-essential python3-dev

RUN pip install cython && pip install scikit-image && pip install django-filter && pip install django-storages
RUN pip install -r requirements.txt 

RUN pip install gast==0.3.3 h5py==2.10.0 

RUN pip uninstall -y numpy
RUN conda install numpy==1.16.1
# --use-feature=2020-resolver
#RUN conda install --yes --file requirements.txt 

#RUN conda deactivate

COPY . /WebDECIMER

#RUN adduser -D user

EXPOSE 8000
ENTRYPOINT ["sh", "entrypoint.sh"]


