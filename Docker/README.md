# ml_lecture

Files used for creation of the Docker image with ML tools:
* TensorFlow
* number of other standard packages

## Creation of image
```bash
sudo docker image build -t candidate  ./ -f Dockerfile
sudo docker image tag candidate akalinow/ml_lecture
sudo docker image push akalinow/ml_lecture
```