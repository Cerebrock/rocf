FROM python:3.6
LABEL author="Matías Grinberg <mtsgrinberg@gmail.com>"

# RUN apt-get update && apt-get upgrade

# RUN mkdir ./app
COPY ./app .

RUN pip3 install -r requirements.txt

CMD python3 main.py
