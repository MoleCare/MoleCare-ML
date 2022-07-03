FROM python:3.8-slim

# Maintainer info
LABEL maintainer="info@molecare.co.uk"
# Make working directories
RUN mkdir -p /molecare-ml-api
WORKDIR ./molecare-ml-api

# Copy application dependencies to the created working directory
COPY Pipfile Pipfile.lock app.py wsgi.py heroku.yml favicon.ico ./
# Copy every file in the source folder to the created working directory
#COPY . .

# Install API dependencies
RUN apt-get update \
    && apt-get -y install curl \
    && apt-get install libgomp1 \
    && pip install --no-cache-dir pipenv \
    && pipenv install --system --deploy --ignore-pipfile

# Change Docker User
# Heroku runs docker apps a a non-root user
RUN adduser --disabled-login molecare-ml-app
USER molecare-ml-app

# Start app
#heroku
CMD gunicorn --bind 0.0.0.0:$PORT wsgi
#wsgi
#CMD gunicorn --bind 0.0.0.0:$PORT app:app
#https://devcenter.heroku.com/articles/container-registry-and-runtime
#https://deparkes.co.uk/2018/03/02/simple-docker-flask-sqlite-api/
# test
# docker run -p 3453:5000 -e PORT=5000 molecare-ml
# docker run -p 5000:5000 -e PORT=5000  molecare-ml
# docker run -p 5001:3453 -e PORT=3453 molecare-ml