https://www.tensorflow.org/install/pip#macos

>python3 --version
Python 3.7.0
>pip3 --version
pip 21.3.1

/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/opt/python/libexec/bin:$PATH"
# if you are on macOS 10.12 (Sierra) use `export PATH="/usr/local/bin:/usr/local/sbin:$PATH"`
brew update
brew install python  # Python 3

>python3 -m venv --system-site-packages ./venv

python3 -m venv --system-site-packages ./venv

>source ./venv/bin/activate  # sh, bash, or zsh
>deactivate #to exit the virtual env

>pip install --upgrade tensorflow #207 MB
>pip install --upgrade pip

Verify the install
>python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"


>
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

try:
  import colab
  !pip install --upgrade pip
except:
  pass

Install TFX
>pip install -U tfx

# Serving is how you apply machine learning model after you’ve trained it
TensorFlow Serving makes the process of taking a model into production easier and faster. It allows you to safely deploy new models and run experiments while keeping the same server architecture and APIs. Out of the box, it provides integration with TensorFlow, but it can be extended to serve other types of models.





