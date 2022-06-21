>export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
>export REPO_NAME=REPOSITORY_NAME
>export IMAGE_NAME=IMAGE_NAME
>export IMAGE_TAG=IMAGE_TAG
>export IMAGE_URI=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

>docker build -f Dockerfile -t ${IMAGE_URI} ./

Verify the container image by running it as a container locally. You likely want to run your training code on a smaller dataset or for a shorter number of iterations than you plan to run on Vertex AI. For example, if the entrypoint script in your container image accepts an --epochs flag to control how many epochs it runs for, you might run the following command:

>docker run ${IMAGE_URI} --epochs 1

If the local run works, you can push the container to Artifact Registry.

First, run gcloud auth configure-docker us-central1-docker.pkg.dev if you have not already done so in your development environment. Then run the following command:

>docker push ${IMAGE_URI}


