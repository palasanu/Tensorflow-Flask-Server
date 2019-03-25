FROM ubuntu

RUN apt-get update

RUN apt-get install sudo

RUN apt-get install python3.6 --assume-yes

RUN echo 'alias python=python3.6' >> ~/.bashrc

RUN apt install curl  --assume-yes

RUN apt-get update && apt-get install -y gnupg

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

RUN apt-get update  --assume-yes

RUN apt-get install tensorflow-model-server  --assume-yes

RUN apt-get upgrade tensorflow-model-server  --assume-yes

RUN sudo apt install python3-pip --assume-yes

RUN echo 'alias pip=pip3' >> ~/.bashrc

COPY requirements.txt /app/requirements.txt

RUN pip3 install -r app/requirements.txt

COPY . ./app

EXPOSE 8000
EXPOSE 8500

CMD ["tensorflow_model_server","--model_base_path=/app/classifier","--rest_api_port=8000", "--model_name=Classifier"]
