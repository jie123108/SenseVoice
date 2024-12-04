FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get clean && apt-get update && apt-get install -y git gcc curl build-essential

ENV TZ=Asia/Shanghai
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata
RUN ls -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y zlib1g-dev openssl libssl-dev libbz2-dev file \
    libexpat1 libmagic-mgc libmagic1 libreadline8 lzma liblzma-dev libbz2-dev \
    libsqlite3-0 libsqlite3-dev libssl1.1 mime-support openssl readline-common xz-utils libffi-dev  \
    && apt-get clean all

RUN curl https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz | tar zxf - && \
    cd ./Python-3.10.13 && \
    ./configure --with-ssl --enable-loadable-sqlite-extensions --enable-optimizations --with-lto --prefix=/usr/local && \
    make -j4 install && \
    cd ../ && \
    rm -rf ./Python-3.10.13 && \
    python3 --version

RUN apt-get update && apt-get install ffmpeg libsndfile1-dev -y
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --no-cache-dir  cython wheel

RUN pip3 install --no-cache-dir torch==2.3.0+cu118 torchaudio==2.3.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install --no-cache-dir  transformers optimum accelerate

RUN pip3 install --no-cache-dir "funasr>=1.1.3" "numpy<=1.26.4" modelscope transformers
WORKDIR /app
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
