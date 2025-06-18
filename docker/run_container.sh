docker rm -f graspmamba
CATGRASP_DIR=$(pwd)/../
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name graspmamba  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --privileged -v /home/:/home/ -v /media/:/media/ -v /etc/udev/rules.d/:/etc/udev/rules.d/ --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE graspmamba:latest bash