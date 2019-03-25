FROM ubuntu

RUN apt-get update

RUN apt-get install sudo

RUN apt-get install python3.6 --assume-yes

RUN echo 'alias python=python3.6' >> ~/.bashrc

RUN apt-get update 

RUN sudo apt install python3-pip --assume-yes

RUN echo 'alias pip=pip3' >> ~/.bashrc

COPY requirements.txt /app/requirements.txt

RUN pip3 install -r app/requirements.txt

COPY . ./app

EXPOSE 8600

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR "/app"

#flask run --host=0.0.0.0 --port=8600
CMD ["flask", "run","--host=0.0.0.0","--port=8600"]