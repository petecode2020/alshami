FROM python:3.10

COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip3 install -r /requirements.txt

COPY . /app
WORKDIR /app

# Make the gunicorn_starter.sh script executable
RUN chmod +x gunicorn_starter.sh

ENTRYPOINT ["./gunicorn_starter.sh"]
