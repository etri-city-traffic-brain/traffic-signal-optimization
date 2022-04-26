#!/bin/bash
##
## generate ssh key and copy to remote host
#        to setup passwordless SSH login for multiple remote servers
##
## you should visit every hosts you want to passwordless SSH login
##      and execute this script
##
## you should check REMOTE_HOSTS variable is valid before exec this script
##

USER_NAME="hunsooni"
REMOTE_HOSTS=(
  "129.254.182.176"
  "129.254.184.53"
)
PUBLIC_KEY_FILE=$HOME/.ssh/id_rsa.pub

###-- generate key
ssh-keygen -t rsa

###-- check if public key file is exist
if [ ! -f  $PUBLIC_KEY_FILE ]; then
        echo "File '$PUBLIC_KEY_FILE' not found!"
        exit 1
fi


###-- copy to remote host
for IP in  ${REMOTE_HOSTS[@]}
do
    ssh-copy-id -i $PUBLIC_KEY_FILE $USER_NAME@$IP
done



#-------------------------------------
## work
## key gen
#-------------------------------------

if [ 1 --eq 0 ]; then
    USER_NAME="hunsooni"

    HOSTS=(
      "129.254.182.176"
      "129.254.184.53"
    )

    ##
    ## create a new SSH key in Linux
    for IP in  ${HOSTS[@]}
    do
      ssh $USER_NAME@$IP ssh-keygen -t rsa
    done
fi

#-------------------------------------
# not work
# key copy
#-------------------------------------
if [ 1 -eq 0 ];  then
    USER_NAME="hunsooni"
    REMOTE_HOSTS=(
      #"129.254.182.176"
      "129.254.184.53"
    )
    PUBLIC_KEY_FILE=$HOME/.ssh/id_rsa.pub
    ERROR_FILE="/tmp/ssh-copy_error.txt"
    LOCAL_IP="129.254.182.176"

    ##-- check if public key file is exist
    #if [ ! -f  $PUBLIC_KEY_FILE ]; then
    #        echo "File '$PUBLIC_KEY_FILE' not found!"
    #        exit 1
    #fi

    for IP in  ${REMOTE_HOSTS[@]}
    do
        echo ssh $USER_NAME@$LOCAL_IP ssh-copy-id -i $PUBLIC_KEY_FILE $USER_NAME@$IP 2>$ERROR_FILE
        ssh $USER_NAME@$LOCAL_IP ssh-copy-id -i $PUBLIC_KEY_FILE $USER_NAME@$IP 2>$ERROR_FILE

        RESULT=$?
        if [ $RESULT -eq 0 ]; then
            echo ""
            echo "Public key successfully copied to $IP"
            echo ""
        else
            echo "$(cat  $ERROR_FILE)"
            echo
            exit 3
        fi
        echo ""
    done
fi