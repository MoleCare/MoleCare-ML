# build the image
docker build -t molecarre-ml .

# run a new docker container named molecare-ml
docker run --name molecare-ml \
    -d -p 5000:5000 \
    cashman

# fetch incomes from the dockerized instance
curl http://localhost:5000/incomes/

The Dockerfile is simple but effective, and using it is similarly easy. With these commands and this Dockerfile, we can run as many instances of our API as we need with no trouble. It's just a matter of defining another port on the host, or even another host.

