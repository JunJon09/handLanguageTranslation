FROM python:3.9

WORKDIR /src

RUN apt-get update

RUN apt-get install -y vim less \
  && pip install --upgrade pip \
  && pip install --upgrade setuptools

# pip で requirements.txt に指定されているパッケージを追加する
COPY requirements.txt ./
RUN pip install -r requirements.txt

CMD [ "bash" ]
