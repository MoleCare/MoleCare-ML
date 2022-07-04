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

# step 7
docker tag molecare-ml:latest 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml-docker-app:latest

# step 8
docker push 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml-docker-app

# step 9
set ec2
>ssh -i molecare-ml-key-pair.pem ec2-user@ec2-YOUR-INSTANCE.compute-1.amazonaws.com

# 10
docker run
> aws ecr describe-repositories
> aws ecr describe-images --repository-name molecare-ml-docker-app 

docker login to ecr repo
> aws ecr get-login-password --region us-east-1 | \
docker login --username AWS --password-stdin \
417382966138.dkr.ecr.us-east-1.amazonaws.com

> docker pull 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml-docker-app:latest
>docker run -d -p 5000:5000 417382966138.dkr.ecr.us-east-1.amazonaws.com/molecare-ml-docker-app:latest
> curl http://localhost:5000/hello/Yauhen



