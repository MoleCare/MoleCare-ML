FROM python:3.8-slim

# Maintainer info
LABEL maintainer="info@molecare.co.uk"
# Make working directories
RUN mkdir -p /molecare-ml-api
WORKDIR ./molecare-ml-api

# Copy application dependencies to the created working directory
COPY Pipfile Pipfile.lock app.py heroku.yml Procfile favicon.ico ./
# Copy every file in the source folder to the created working directory
#COPY . .

# Install API dependencies
RUN apt-get update \
    && apt-get -y install curl \
    && apt-get install libgomp1
RUN pip install --no-cache-dir pipenv
RUN pipenv install --system --deploy --ignore-pipfile

# Start app
#heroku
CMD gunicorn --bind 0.0.0.0:$PORT app:app
# test
# docker run -p 3453:5000 -e PORT=5000 molecare-ml
# docker run -p 5001:3453 -e PORT=3453 molecare-ml