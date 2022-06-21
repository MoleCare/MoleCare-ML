#!/bin/sh
# build the image
docker image build -t molecare-ml .

# run a new docker container named molecare-ml
docker run --name molecare-ml \
    -d -p 5000:5000 \
    molecare-ml

# fetch hello from the dockerized instance
curl http://localhost:5000/
