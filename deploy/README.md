https://devcenter.heroku.com/articles/container-registry-and-runtime

https://help.heroku.com/PPBPA231/how-do-i-use-the-port-environment-variable-in-container-based-apps

To push multiple images, rename your Dockerfiles using Dockerfile.<process-type>:

ls -R

./webapp:
Dockerfile.web

./worker:
Dockerfile.worker

./image-processor:
Dockerfile.image
Then, from the root directory of the project, run:

heroku container:push --recursive
=== Building web
=== Building worker
=== Building image
=== Pushing web
=== Pushing worker
=== Pushing image

This will build and push all 3 images. If you only want to push specific images, you can specify the process types:

heroku container:push web worker --recursive
=== Building web
=== Building worker
=== Pushing web
=== Pushing worker

After you’ve successfully pushed an image to Container Registry, you can create a new release using:

heroku container:release web
If you have multiple images, list them:

heroku container:release web worker

