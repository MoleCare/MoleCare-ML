pip install mod-wsgi-httpd
pip install mod-wsgi
pip install gunicorn

pipenv shell
gunicorn --bind 0.0.0.0:5000 --timeout 300 wsgi
ps ax|grep gunicorn
kill -9 7926

> >docker-compose up
>http://localhost:1337/

> docker build -t molecare-ml .
available through localhost:5001
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
>source ./venv/bin/activate
>python3 -m flask run
> 
>python -m flask run
>gunicorn --bind 0.0.0.0:5000 --timeout 300 wsgi

# Install tensor flow serving
https://www.tensorflow.org/tfx/serving/docker

>docker pull tensorflow/serving
>tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}  ex: tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=/models/model

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

# AWS
https://aws.amazon.com/blogs/opensource/deploying-python-flask-microservices-to-aws-using-open-source-tools/

# step 1
aws ecr create-repository \
--repository-name molecare-ml-docker-app \
--image-scanning-configuration scanOnPush=true \
--region us-east-1

## response
    {
        "repository": {
            "repositoryArn": "arn:aws:ecr:us-east-1:417382966138:repository/molecare-ml-docker-app",
            "registryId": "417382966138",
            "repositoryName": "molecare-ml-docker-app",
            "repositoryUri": "417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml-docker-app",
            "createdAt": "2022-07-04T08:13:28+00:00",
            "imageTagMutability": "MUTABLE",
            "imageScanningConfiguration": {
                "scanOnPush": true
            },
            "encryptionConfiguration": {
                "encryptionType": "AES256"
            }
        }
    }

# step 2
install AWS CLI
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

# step 3
configure aws credentials
https://aws.amazon.com/premiumsupport/knowledge-center/s3-locate-credentials-error/

>aws configure list
> aws configure --profile AWS

AWSAccessKeyId=***REMOVED-AWS-ACCESS-KEY***
AWSSecretKey=***REMOVED-AWS-SECRET-KEY***
region=us-east-1
format=text

# step 5 aws cli login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml-docker-app

# step 6
>docker build --tag molecare-ml .

>docker context use default
>docker build --tag molecare-ml_tensorflow-serving --file Dockerfile.serving .
>docker build --tag molecare-ml_nginx --file Dockerfile.nginx .
>docker build --tag molecare-ml_web --file Dockerfile.web .

# step 7
docker tag molecare-ml:latest 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml-docker-app:latest

>docker tag molecare-ml_tensorflow-serving:latest 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml_tensorflow-serving:latest
>docker tag molecare-ml_nginx:latest 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml_nginx:latest
>docker tag molecare-ml_web:latest 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml_web:latest

# step 8
docker push 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml-docker-app
## ecr should have the repos as molecare-ml_tensorflow-serving
>docker push 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml_tensorflow-serving
>docker push 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml_nginx
>docker push 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml_web

# step 9
set ec2
>chmod 700 molecare-ml-key-pair.pem
>ssh -i molecare-ml-key-pair.pem ec2-user@ec2-YOUR-INSTANCE.compute-1.amazonaws.com

# 10
docker run
> aws ecr describe-repositories
> aws ecr describe-images --repository-name molecare-ml-docker-app

>sudo chmode 777 .
>sudo scp -i molecare-ml-key-pair.pem ~/DevBox/MoleCare-app/MoleCare-ML/*  ec2-user@ec2-YOUR-INSTANCE.compute-1.amazonaws.com:/
>scp -i molecare-ml-key-pair.pem ~/DevBox/MoleCare-app/MoleCare-ML/*  ec2-user@ec2-YOUR-INSTANCE.compute-1.amazonaws.com:/

docker login to ecr repo
> aws ecr get-login-password --region us-east-1 | \
docker login --username AWS --password-stdin \
417382966138.dkr.ecr.us-east-1.amazonaws.com

> docker pull 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml-docker-app:latest
>docker run -d -p 5000:5000 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml-docker-app:latest
> curl http://localhost:5000/hello/Yauhen

## docker compose
>docker compose build ml-docker-compose
>docker tag ml-docker-compose:latest 417382966138.dkr.ecr.us-east-1.amazonaws.com/ml-docker-compose:latest
docker compose push 417382966138.dkr.ecr.us-east-1.amazonaws.com/ml-docker-compose
docker-compose push 417382966138.dkr.ecr.us-east-1.amazonaws.com/ml-docker-compose