# docker run -it --gpus all -v $PATH_TO_REPO:/storage -v $PATH_TO_VIDEO_ROOT_FOLDER:/scratch2 porsche /bin/bash

CONTAINER_ID=$(docker ps -alq)
docker start -ai $CONTAINER_ID
