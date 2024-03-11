FROM python:3.10-bullseye

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
COPY requirements.txt ./

RUN pip3 install --no-cache -r requirements.txt

COPY . ./