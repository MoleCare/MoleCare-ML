FROM python:3.7.3-stretch

# Maintainer info
LABEL maintainer="info@molecare.co.uk"

# Make working directories
RUN  mkdir -p  /molecare-ml-api
WORKDIR  /molecare-ml-api

# Copy application dependencies to the created working directory
COPY Pipfile Pipfile.lock bootstrap-dev.sh ./

# Install API dependencies
RUN pip install --no-cache-dir pipenv

# Copy every file in the source folder to the created working directory
COPY  . .

# Run the python application
CMD ["python", "app.py"]

# Start app
EXPOSE 5000
ENTRYPOINT ["bootstrap-dev.sh"]
