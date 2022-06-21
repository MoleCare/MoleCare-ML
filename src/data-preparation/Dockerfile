# syntax=docker/dockerfile:1
# FROM creates a layer from the ubuntu:18.04 Docker image.
# COPY adds files from your Docker client’s current directory.
# RUN builds your application with make.
# CMD specifies what command to run within the container.
#-----

# Specifies base image and tag
FROM image:tag
WORKDIR /root

# Installs additional packages
RUN pip install pkg1 pkg2 pkg3

# Downloads training data
RUN curl https://example-url/path-to-data/data-filename --output /root/data-filename

# Copies the trainer code to the docker image.
COPY your-path-to/model.py /root/model.py
COPY your-path-to/task.py /root/task.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "task.py"]


