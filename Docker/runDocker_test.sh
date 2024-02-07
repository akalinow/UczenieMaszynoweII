sudo docker run --rm -it -p 9000:9000 --user $(id -u):$(id -g) -v /scratch_hdd:/scratch_hdd -v /scratch_ssd:/scratch_ssd -v /scratch_cmsse:/scratch_cmsse --gpus all --cap-add=CAP_SYS_ADMIN candidate

# --user $(id -u):$(id -g)