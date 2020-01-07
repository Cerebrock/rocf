FROM tiangolo/uwsgi-nginx-flask:python3.6

COPY ./app /app
COPY nginx /etc/nginx

RUN pip install -r requirements.txt
