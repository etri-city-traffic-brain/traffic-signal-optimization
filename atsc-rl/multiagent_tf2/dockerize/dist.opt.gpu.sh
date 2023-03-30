#!/bin/sh
#
# [%] dist.opt.gpu.sh version# dockerfile_name image_name copy_binary
#
#    assumption : binaries are located in $SALT_HOME and $OPT_HOME
#    SALT_HOME=/home/developer/uniq/traffic-simulator
#    OPT_HOME=/home/developer/uniq/traffic-signal-optimization
#
#    sholud check the location of binaries to dockerize
#    should set DO_PUSH if you want to push into docker repository

DO_PUSH=false # whether do push into docker repository or not : do push if true, otherwise skip push

DEFAULT_VERSION="v2.10.0-gpu" # "v0.0a"   # "v2.1a.0622"
VERSION=${1:-$DEFAULT_VERSION}

DEFAULT_DOCKERFILE="Dockerfile.opt.gpu" # compile within docker env
DOCKERFILE=${2:-$DEFAULT_DOCKERFILE}

DEFAULT_IMAGE_NAME="optimizer"
IMAGE_NAME=${3:-$DEFAULT_IMAGE_NAME}

#REPO_ID="hunsooni" 	#repository id  # easy one + #
REPO_ID="images4uniq"	#repository id  # '21 project account number + #

DEFAULT_COPY_BINARY="yes"
COPY_BINARY=${4:-$DEFAULT_COPY_BINARY}



# O. copy source code
if [ "$COPY_BINARY" = "yes" ]; then
    # 0.0 set source directory
    SALT_HOME=/home/tsoexp/z.docker_test/traffic-simulator
    OPT_HOME=/home/tsoexp/z.docker_test/io/multiagent_tf2.yjlee

    # 0.1 remove old codes & create empty directory
    # 0.1.1 remove old binary
    rm -rf ./to_install_uniq/salt
    rm -rf ./to_install_uniq/optimizer

    # 0.1.2 create empty directory
    mkdir ./to_install_uniq/salt
    mkdir ./to_install_uniq/optimizer

    # 0.2 copy SALT source code
    cp -r $SALT_HOME/* ./to_install_uniq/salt

    # 0.3 copy OPTIMIZER source code
    cp -r $OPT_HOME/*.py ./to_install_uniq/optimizer
    cp -r $OPT_HOME/README.md ./to_install_uniq/optimizer
    cp -r $OPT_HOME/env ./to_install_uniq/optimizer
    cp -r $OPT_HOME/policy ./to_install_uniq/optimizer

    cp -r $OPT_HOME/README_DIST.md ./to_install_uniq/optimizer
    cp -r $OPT_HOME/dist_training.sh ./to_install_uniq/optimizer
    cp -r $OPT_HOME/sshKeyGenAndCopy.sh ./to_install_uniq/optimizer

fi

#
#
# 1. build docker image 
#
echo "[%] sudo docker build -f $DOCKERFILE -t $IMAGE_NAME:$VERSION ."
sudo docker build -f $DOCKERFILE -t $IMAGE_NAME:$VERSION .


#
# you can test built docker image is correctly work
#
# sudo docker run -v /home/hunsooni/z.uniq/simulator/traffic-simulator/dockerize/volume:/uniq/simulator/salt/volume hunsooni/salt:v0.1a ./bin/salt-standalone ./volume/sample/sample.json
#echo sudo docker run  -v ${PWD}/volume:/uniq/simulator/salt/volume $IMAGE_NAME:$VERSION ./bin/salt-standalone ./volume/sample/sample.json
#sudo docker run  -v ${PWD}/volume:/uniq/simulator/salt/volume $IMAGE_NAME:$VERSION ./bin/salt-standalone ./volume/sample/sample.json

#
# 2. tag built docker image
#
#sudo docker tag salt:v2.1a.0622 images4uniq/salt:v2.1a.0622
echo [%] sudo docker tag $IMAGE_NAME:$VERSION $REPO_ID/$IMAGE_NAME:$VERSION
sudo docker tag $IMAGE_NAME:$VERSION $REPO_ID/$IMAGE_NAME:$VERSION

#
# 3. check built/tagged image is exist
#
echo [%] sudo docker images
sudo docker images


#
# 4. login docker repository
#
if $DO_PUSH  # check whether do push into docker repository or not
then
    echo [%] sudo docker login $REPO_ID
    sudo docker login -u $REPO_ID
fi
#
# 5. push docker image into repository
#
if $DO_PUSH # check whether do push into docker repository or not
then
    # sudo docker push images4uniq/salt:v2.1a.0622
    echo [%] sudo docker push $REPO_ID/$IMAGE_NAME:$VERSION
    sudo docker push $REPO_ID/$IMAGE_NAME:$VERSION
fi
