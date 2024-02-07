sudo docker run --rm -it -p 8008:8008 -p 8888:8888 --user $(id -u):$(id -g) -v /scratch_hdd:/scratch_hdd -v /scratch_ssd:/scratch_ssd -v /scratch_cmsse:/scratch_cmsse --gpus all --cap-add=CAP_SYS_ADMIN akalinow/ml_lecture

