FROM floydhub/tensorflow:1.7.0-py3_aws.25
MAINTAINER Floyd Labs "support@floydhub.com"

RUN pip --no-cache-dir install --upgrade http://download.pytorch.org/whl/cu91/torch-0.3.1-cp36-cp36m-linux_x86_64.whl \
    tensorboardX \
    torchvision==0.2.0
RUN pip --no-cache-dir install --upgrade awscli
RUN pip --no-cache-dir install --upgrade boto3
RUN pip --no-cache-dir install --upgrade mypy

# Install utilities for Coco dataset.
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /cocoapi/PythonAPI
RUN make
RUN make install
WORKDIR /

# Core Python library
ADD cse547 /tmp/cse547/
ADD setup.py /tmp/
WORKDIR tmp
RUN pip --no-cache-dir install --upgrade .
WORKDIR /
RUN rm -rf /tmp/cse547
RUN rm -rf /tmp/setup.py

# Python binaries
ADD hw1/run.py /hw1/
RUN chmod +x /hw1/run.py
