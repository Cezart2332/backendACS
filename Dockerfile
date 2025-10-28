FROM python:3.15.0a1-slim-trixie

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD flask --app backend run --host=0.0.0.0
