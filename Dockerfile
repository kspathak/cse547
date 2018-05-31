FROM floydhub/tensorflow:1.8.0-py3_aws.28
MAINTAINER Floyd Labs "support@floydhub.com"

RUN pip --no-cache-dir install --upgrade http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl torchvision==0.2.1
RUN pip --no-cache-dir install --upgrade tensorboardX==1.2
RUN pip --no-cache-dir install --upgrade awscli
RUN pip --no-cache-dir install --upgrade boto3
RUN pip --no-cache-dir install --upgrade joblib
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
ADD hw2/run.py /hw2/
RUN chmod +x /hw2/run.py
ADD hw3/run.py /hw3/
RUN chmod +x /hw3/run.py
ADD hw3/generate_examples.py /hw3/
RUN chmod +x /hw3/generate_examples.py
ADD project/embed.py /project/
ADD project/run.py /project/
RUN chmod +x /project/embed.py
RUN chmod +x /project/run.py
