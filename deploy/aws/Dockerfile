FROM python:3.7.3-slim

# Maintainer info
LABEL maintainer="info@molecare.co.uk"

# Make working directories
RUN mkdir -p /molecare-ml-api
WORKDIR ./molecare-ml-api

# Copy application dependencies to the created working directory
COPY Pipfile Pipfile.lock bootstrap-heroku.sh app.py main.py heroku.yml Procfile ./
# Copy every file in the source folder to the created working directory
#COPY . .

# Install API dependencies
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
    && apt-get -y install curl \
    && apt-get install libgomp1
RUN pip install --no-cache-dir pipenv
RUN pipenv install --system --deploy --ignore-pipfile

# Start app
#localhost
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

#ENTRYPOINT ["bootstrap-dev.sh"]
#CMD gunicorn --bind 0.0.0.0:$PORT wsgi
#CMD gunicorn app:server --bind 0.0.0.0:$PORT --preload
#CMD gunicorn --bind 0.0.0.0:$PORT
#CMD gunicorn main:app --bind 0.0.0.0:5000
