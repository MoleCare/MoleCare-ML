FROM python:3.7.3-alpine

# Maintainer info
LABEL maintainer="info@molecare.co.uk"

# Make working directories
RUN mkdir -p /molecare-ml-api
WORKDIR ./molecare-ml-api

# Copy application dependencies to the created working directory
#COPY Pipfile Pipfile.lock bootstrap-dev.sh ./
# Copy every file in the source folder to the created working directory
COPY . .

# Install API dependencies
#RUN apt -y update
#RUN apt -y install curl
RUN pip install --no-cache-dir pipenv
RUN pipenv install

# Start app
EXPOSE 5000
#ENTRYPOINT ["bootstrap-dev.sh"]
#CMD sh bootstrap-dev.sh
CMD python -m flask run