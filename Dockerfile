FROM python:3.7-slim
RUN apt-get update
RUN apt-get -y install git
RUN apt-get -y install protobuf-compiler
# install modules
RUN pip install -q Cython
RUN pip install -q contextlib2
RUN pip install -q pillow
RUN pip install -q lxml
RUN pip install tensorflow
# RUN pip install -q jupyter
RUN pip install -q flask

RUN mkdir /pydir
WORKDIR /pydir
RUN git clone https://github.com/tensorflow/models.git
WORKDIR models/research
RUN protoc object_detection/protos/*.proto --python_out=.
#CHANGE after test change to git clone
WORKDIR /pydir
RUN git clone https://github.com/Blajimir/rest-service-for-od-tf-api.git
#RUN mkdir rest-service-for-od-tf-api
WORKDIR rest-service-for-od-tf-api/
#COPY ./restapi.py /pydir/rest-service-for-od-tf-api/
#COPY ./loadmodels.py /pydir/rest-service-for-od-tf-api/
#COPY ./model-list.json /pydir/rest-service-for-od-tf-api/
#TODO: Delete after test
#COPY ./model-list-test.json /pydir/rest-service-for-od-tf-api/
#CHANGE end
RUN python loadmodels.py
CMD python restapi.py
#ENV FLASK_APP=restapi.py
#CMD flask run
EXPOSE 5000