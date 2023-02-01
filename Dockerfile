# ARG FROM_IMAGE_NAME=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git vim wget

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt

# RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && \
#     HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_PYTORCH=1 \
#     pip install --no-cache-dir --upgrade --force-reinstall horovod[pytorch] && ldconfig

RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod

#RUN pip uninstall -y apex; pip uninstall -y apex; git clone --branch seryilmaz/fused_dropout_softmax  https://github.com/seryilmaz/apex.git; cd apex;  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--xentropy" --global-option="--deprecated_fused_adam" --global-option="--deprecated_fused_lamb" --global-option="--fast_multihead_attn" .
