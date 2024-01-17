ARG FROM_IMAGE_NAME=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
FROM ${FROM_IMAGE_NAME}

# Copy the application code
ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

# Install additional dependencies and clean up APT cache
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install -y vim

# Upgrade pip and install requirements
RUN pip install --upgrade pip
RUN pip install memory-profiler

RUN pip install --disable-pip-version-check -r requirements.txt

#COPY preprocess.py .


# Copy the startup script
#COPY start_training.sh /workspace/unet3d/start_training.sh
#RUN chmod +x /workspace/unet3d/start_training.sh

# Run the startup script in privileged mode
#CMD ["/workspace/unet3d/start_training.sh"]
