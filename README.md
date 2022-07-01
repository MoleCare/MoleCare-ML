> gunicorn --bind 0.0.0.0:5000 main:app
> docker build -t molecare-ml .
> docker run -d -p 5001:5000 molecare-ml
> docker ps
> docker inspect <container id>
    > docker inspect 74cffd781b1b | grep "IPAddress"
> docker inspect 4e77bca22cb3 | grep "IPAddress"
> 172.17.0.0/16
> 192.168.65.0
> kill -9 $(lsof -ti:5000)

# Run
>docker image build -t molecare-ml .
>docker run -p 5000:5000 -e PORT=5000 -d molecare-ml
>docker-build-run.sh

# Stop
>docker container stop <container-id>
>docker system prune

# Heroku https://dashboard.heroku.com/apps/molecare-ml-api
# https://molecare-ml-api.herokuapp.com/
> heroku login
> heroku container:login
> heroku container:push molecare-ml --app molecare-ml-api
> heroku container:release molecare-ml --app molecare-ml-api

> heroku logs --tail --app molecare-ml-api



# Package manager
## pipenv
Pipenv is a dependency manager that isolates projects on private environments, allowing packages to be installed per project.
>python3 -m pip install pipenv
>pipenv install requests

Install from Pipfile, if there is one:
>pipenv install

Or, add a package to your new project:
>pipenv install <package>

>pipenv update
>pipenv update --outdated

### using lock
>pipenv lock
>pipenv lock --keep-outdated


# MoleCare Melanoma CNN model with Rest API
>python --version
>pip --version
>curl https://bootstrap.pypa.io/pip/2.7/get-pip.py | python
>curl https://bootstrap.pypa.io/get-pip.py | python
>curl https://bootstrap.pypa.io/get-pip.py | python
>alias pip=pip3


# Create environment macOS
>python3 -m venv .venv
>source .venv/bin/activate

# How to run flask
>source ./venv/bin/activate  # sh, bash, or zsh
>python3 -m flask run
>python -m flask run

# Install tensor flow serving
https://www.tensorflow.org/tfx/serving/docker

>docker pull tensorflow/serving
>tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}  ex: tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=/models/model

tensorflow_model_server --model_base_path=/home/ubuntu/Desktop/Medium/keras-and-tensorflow-serving/my_image_classifier --rest_api_port=9000 --model_name=ImageClassifier

--rest_api_port: Tensorflow Serving will start a gRPC ModelServer on port 8500 and the REST API will be available on port 9000.
--model_name: This will be the name of your Serving server using which you will send a POST request. You can type any name you want here.


The script basically mimics request from the frontend:

We take an input image, encode it to base64 format and send it to our Flask server using POST request.
Flask server decodes this base64 image and pre-processes it for our TensorFlow Serving server.
Flask server then makes a POST request to our TensorFlow serving server and decodes the response.
The decoded response is formatted and sent back to the frontend.

# CORS
https://flask-cors.readthedocs.io/en/latest/index.html
>pip install -U flask-cors
> 

# Package manager
## pipenv
Pipenv is a dependency manager that isolates projects on private environments, allowing packages to be installed per project.
>python3 -m pip install pipenv
>pipenv install requests

Install from Pipfile, if there is one:
>pipenv install

Or, add a package to your new project:
>pipenv install <package>

>pipenv update
>pipenv update --outdated

### using lock
>pipenv lock
>pipenv lock --keep-outdated

## migrate from requirements.txt to pipenv
>pipenv install -r requirements.txt
## pip requirements.txt
pip supports package management through the requirements.txt file
> pip install -e .
> setup.py # contains dependencies, which are installed by cmd `pip install -e .`
> pip list # observe that the project is now installed with pip list

>pip freeze > requirements.txt
>pip install -r requirements.txt


https://packaging.python.org/en/latest/tutorials/packaging-projects/
# Build
>python3 -m pip install --upgrade build
>python3 -m build


# Django vs Flask
https://auth0.com/blog/developing-restful-apis-with-python-and-flask/

# Run tests
>pytest
>coverage run -m pytest
>coverage report
>coverage html
>py.test tests.py --cov=molecare-ml

# Github
>git push molecare main (molecare is origin repo, and main is master branch)


export FLASK_APP=molecare-ml.application
export FLASK_ENV=development
flask run



# Python distribution format is wheel with the .whl extension.
>python setup.py bdist_wheel


# Database
Initialized the database.
>flask init-db


# Google Cloud Storage
>pipenv install google-cloud-storage

# Heroku
- setup.sh: this file is necessary to handle the server and port number of our app on Heroku.
- Procfile: this is the file of your configuration to tell Heroku how and which files to be executed




