FROM ubuntu:20.04

RUN apt-get update
RUN DEBIAN_FRONTEND='noninteractive' apt-get install -y python3-pip
RUN DEBIAN_FRONTEND='noninteractive' apt install unixodbc-dev -y
RUN DEBIAN_FRONTEND='noninteractive' apt-get install libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev -y
RUN DEBIAN_FRONTEND='noninteractive' apt-get update && apt-get install -y gcc
RUN pip3 install --upgrade pip
RUN pip3 install cython

RUN pip3 install --no-cache-dir Flask==1.0.2
RUN pip3 install --no-cache-dir Flask-Cors==3.0.10
RUN pip3 install --no-cache-dir numpy==1.19.1
RUN pip3 install --no-cache-dir pandas==1.1.5
RUN pip3 install --no-cache-dir scipy==1.4.1
RUN pip3 install --no-cache-dir requests==2.26.0

RUN pip3 install --no-cache-dir nltk==3.5
RUN pip3 install --no-cache-dir sklearn==0.0

RUN pip3 install --no-cache-dir torch==1.6.0
RUN pip3 install --no-cache-dir sentence-transformers==2.1.0

RUN pip3 install Werkzeug==2.0.0
RUN pip3 install jinja2==3.0

WORKDIR app
COPY . .

EXPOSE 5001

CMD ["python3", "app.py"]