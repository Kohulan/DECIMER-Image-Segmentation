FROM python:3
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt install -y libgl1-mesa-glx poppler-utils

WORKDIR /WebDECIMER
COPY auto.requirements.txt /WebDECIMER

RUN pip install -r auto.requirements.txt

COPY . /WebDECIMER

EXPOSE 8000
ENTRYPOINT ["sh", "entrypoint.sh"]


