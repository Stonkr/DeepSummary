FROM python:3.9.13-slim
RUN apt-get update

RUN DEBIAN_FRONTEND='noninteractive' apt install unixodbc-dev -y
RUN DEBIAN_FRONTEND='noninteractive' apt-get install libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev -y
RUN DEBIAN_FRONTEND='noninteractive' apt-get update && apt-get install -y gcc

WORKDIR summary_gen
COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install cython
RUN pip3 install -r requirements.txt

RUN python3 -m nltk.downloader punkt

EXPOSE 5001

CMD ["python3", "main.py"]