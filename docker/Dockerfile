FROM python:3.9

WORKDIR /src

RUN apt-get update

# OpenCVのGUI表示に必要なパッケージを追加
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    libxcb1 libxcb-shm0 libxcb-xfixes0 libxcb-shape0 libxcb-randr0 \
    libxcb-image0 libxcb-keysyms1 libxcb-icccm4 libxcb-sync1 \
    libxcb-xtest0 libxcb-xinerama0 x11-utils

RUN apt-get install -y vim less \
  && pip install --upgrade pip \
  && pip install --upgrade setuptools
  
RUN apt-get install -y fonts-noto-cjk


COPY requirements.txt ./
RUN pip install -r requirements.txt

CMD [ "bash" ]