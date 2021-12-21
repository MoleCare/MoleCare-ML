# flask depends on this env variable to find the main file
export FLASK_APP=hello.py

# now we just need to ask flask to run
flask run

# Virtual Environments (virtualenv)
pip installs packages globally, making it hard to manage multiple versions of the same package on the same machine.
requirements.txt need all dependencies and sub-dependencies listed explicitly, a manual process that is tedious and error-prone.
To solve these issues, we are going to use Pipenv. Pipenv is a dependency manager that isolates projects on private environments, allowing packages to be installed per project.
> pip3 install pipenv
> pipenv install flask


