>python3 -m venv venv
>source .venv/bin/activate


# MoleCare Melanoma CNN model with Rest API

>python --version
>pip --version
>curl https://bootstrap.pypa.io/pip/2.7/get-pip.py | python
>curl https://bootstrap.pypa.io/get-pip.py | python
>curl https://bootstrap.pypa.io/get-pip.py | python
>alias pip=pip3


# Create environment macOS
python3 -m venv .venv
source .venv/bin/activate

>python3 -m venv .venv
>source .venv/bin/activate

>pip3 install matplotlib
>pip3 install imageai
>pip3 install tensorflow
>pip3 install pillow
>pip3 install numpy
>pip3 install opencv-python

>/Applications/Xcode.app/Contents/Developer/usr/bin/python3 -m pip install --upgrade pip
>pip3 install flask


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
## pip requirements.txt
pip supports package management through the requirements.txt file
> pip install -e .
> setup.py # contains dependencies, which are installed by cmd `pip install -e .`
> pip list # observe that the project is now installed with pip list

>pip freeze > requirements.txt
>pip install -r requirements.txt

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

https://packaging.python.org/en/latest/tutorials/packaging-projects/
# Build
>python3 -m pip install --upgrade build
>python3 -m build

#
- install is Python 3, Pip (Python Package Index), and Flask
- 


# Django vs Flask
https://auth0.com/blog/developing-restful-apis-with-python-and-flask/

# Run tests
>pytest
>coverage run -m pytest
>coverage report
>coverage html

# Github
>git push molecare main (molecare is origin repo, and main is master branch)


export FLASK_APP=molecare-ml.application
export FLASK_ENV=development
flask run

