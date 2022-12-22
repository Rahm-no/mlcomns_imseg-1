ARG FROM_IMAGE_NAME=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git vim wget

# apt-get install -y libtool autoconf build-essential
# # Install Darshan
# WORKDIR /opt
# RUN wget https://ftp.mcs.anl.gov/pub/darshan/releases/darshan-3.4.0.tar.gz
# RUN tar -xzf darshan-3.4.0.tar.gz
# WORKDIR darshan-3.4.0
# RUN ./prepare.sh
# WORKDIR darshan-runtime/
# RUN ./configure --with-log-path-by-env=DARSHAN_LOGPATH --with-jobid-env=NONE --without-mpi --enable-mmap-logs --enable-group-readable-logs CC=gcc
# RUN make
# RUN make install

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt

#RUN pip uninstall -y apex; pip uninstall -y apex; git clone --branch seryilmaz/fused_dropout_softmax  https://github.com/seryilmaz/apex.git; cd apex;  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--xentropy" --global-option="--deprecated_fused_adam" --global-option="--deprecated_fused_lamb" --global-option="--fast_multihead_attn" .
