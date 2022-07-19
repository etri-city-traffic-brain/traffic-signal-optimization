#!/bin/sh
#
# [%] dist.salt.sh version# dockerfile_name image_name copy_binary
#
#    assumption : binaries are located in $SALT_HOME and $OPT_HOME
#    SALT_HOME=/home/developer/uniq/traffic-simulator
#    OPT_HOME=/home/developer/uniq/traffic-signal-optimization
#
#    sholud check the location of binaries to dockerize

DEFAULT_VERSION="v1.1a.20220531" # "v0.0a"   # "v2.1a.0622"
VERSION=${1:-$DEFAULT_VERSION}

DEFAULT_DOCKERFILE="Dockerfile.opt.compile" # compile within docker env
DOCKERFILE=${2:-$DEFAULT_DOCKERFILE}

DEFAULT_IMAGE_NAME="optimizer"
IMAGE_NAME=${3:-$DEFAULT_IMAGE_NAME}

#REPO_ID="hunsooni" 	#repository id  # easy one + #
REPO_ID="images4uniq"	#repository id  # '21 project account number + #

DEFAULT_COPY_BINARY="yes"
COPY_BINARY=${4:-$DEFAULT_COPY_BINARY}

# O. copy binaries
if [ "$COPY_BINARY" = "yes" ]; then
    # 0.0 set source directory
    #SALT_HOME=/home/developer/PycharmProjects/z.uniq/traffic-simulator
    #SALT_HOME=/home/developer/PycharmProjects/z.uniq/traffic-simulator-test_BUS
    #SALT_HOME=/home/pi/traffic-simulator
    SALT_HOME=/home/tsoexp/z.docker_test/traffic-simulator
    #OPT_HOME=/home/developer/PycharmProjects/z.uniq/traffic-signal-optimization/salt-rl
    #OPT_HOME=/home/developer/PycharmProjects/uniq.dockerize/dockerize.opt/to_install_uniq/optimizer_home/
    #OPT_HOME=/home/developer/PycharmProjects/z.uniq/traffic-signal-optimization/salt-rl
    #OPT_HOME=/home/pi/etri-city-traffic-brain/traffic-signal-optimization/atsc-rl/multi-agent
    OPT_HOME=/home/tsoexp/PycharmProjects/traffic-signal-optimization-for-dist/atsc-rl/multiagent_tf2.20220530PM

    # 0.1 remove old binary & create empty directory
    # 0.1.1 remove old binary
    rm -rf ./to_install_uniq/salt
    rm -rf ./to_install_uniq/optimizer

    # 0.1.2 create empty directory
    mkdir ./to_install_uniq/salt
    mkdir ./to_install_uniq/optimizer

    # 0.2 copy SALT source
    cp -r $SALT_HOME/* ./to_install_uniq/salt
    # cp -r $SALT_HOME/bin ./to_install_uniq/salt
    # cp -r $SALT_HOME/tools ./to_install_uniq/salt
    # cp -r ./to_install_uniq/additional/salt_data/salt*.* ./to_install_uniq/salt/bin
    # cp -r ./to_install_uniq/additional/salt_data/sample ./to_install_uniq/salt

    # 0.3 copy OPTIMIZER binary 
    cp -r $OPT_HOME/*.py ./to_install_uniq/optimizer
    cp -r $OPT_HOME/README.md ./to_install_uniq/optimizer
    cp -r $OPT_HOME/env ./to_install_uniq/optimizer
    cp -r $OPT_HOME/policy ./to_install_uniq/optimizer
    #cp -r ./to_install_uniq/additional/opt_data/magic ./to_install_uniq/optimizer

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
echo [%] sudo docker login $REPO_ID
sudo docker login -u $REPO_ID

#
# 5. push docker image into repository
#
# sudo docker push images4uniq/salt:v2.1a.0622
echo [%] sudo docker push $REPO_ID/$IMAGE_NAME:$VERSION
sudo docker push $REPO_ID/$IMAGE_NAME:$VERSION
