FROM python:3.7.3-slim

# Maintainer info
LABEL maintainer="info@molecare.co.uk"

ARG port
ENV PORT=$port

# Make working directories
RUN mkdir -p /molecare-ml-api
WORKDIR ./molecare-ml-api

# Copy application dependencies to the created working directory
COPY Pipfile Pipfile.lock bootstrap-heroku.sh app.py heroku.yml ./
# Copy every file in the source folder to the created working directory
#COPY . .

# Install API dependencies
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
    && apt-get -y install curl \
    && apt-get install libgomp1
RUN pip install --no-cache-dir pipenv
RUN pipenv install --system --deploy --ignore-pipfile

# Start app
EXPOSE $PORT
#ENTRYPOINT ["bootstrap-dev.sh"]
#RUN sh bootstrap-heroku.sh
#CMD gunicorn --bind 0.0.0.0:$PORT wsgi
#CMD gunicorn app:server --bind 0.0.0.0:$PORT --preload
CMD sh bootstrap-heroku.sh
