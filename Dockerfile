FROM python:3.9.13-slim
ENV INTELLIGENCE_HOST=intelligence

RUN apt-get update
RUN DEBIAN_FRONTEND='noninteractive' apt install unixodbc-dev -y
RUN DEBIAN_FRONTEND='noninteractive' apt-get install libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev -y
RUN DEBIAN_FRONTEND='noninteractive' apt-get update && apt-get install -y gcc
RUN pip3 install --upgrade pip
RUN pip3 install cython


WORKDIR summary_gen
COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install cython
RUN pip3 install --no-cache Flask==1.0.2
RUN pip3 install --no-cache Flask-Cors==3.0.10
RUN pip3 install --no-cache Werkzeug==2.0.0
RUN pip3 install --no-cache jinja2==3.0
RUN pip3 install --no-cache pandas==1.1.5
RUN pip3 install --no-cache requests==2.26.0
RUN pip3 install --no-cache nltk==3.5
RUN pip3 install --no-cache sklearn==0.0
RUN pip3 install --no-cache itsdangerous==2.0.1

RUN python3 -m nltk.downloader punkt

EXPOSE 5001

CMD ["python3", "main.py"]
