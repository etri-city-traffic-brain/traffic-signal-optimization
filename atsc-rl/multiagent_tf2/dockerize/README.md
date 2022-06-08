
## How to distribute UNIQ binaries using docker image.

launch distribution script(dist.opt.sh, dist.salt.sh) after read followings

### should check
* default values of variables in shell script(dist.*.sh) 
  * DEFAULT_VERSION : version of this distribution
  * DEFAULT_DOCKERFILE : dockerfile which  is a text document that contains all the commands a user could call on the command line to assemble an image.
  * REPO_ID : id which will be used to login docker repository
     * images4uniq : official ID for UNIQ
  * DEFAULT_COPY_BINARY : whether get a new version of binary or not
  * SALT_HOME : the location of binary for simulator to distribute
  * OPT_HOME : the location of binary for optimizer to distribute
   
* do not forget checking the location of binaries to distribute

   
<hr>

### files
* Dockerfile  
 a text document that contains all the commands a user could call on the command line to assemble an image.
  * Dockerfile.salt
    * script to build docker image for simulator
  * Dockerfile.opt
    * script to build docker image for optimizer
      * contains SALT simulator
  
  
* distribution file  
scripts for automation : docker image build, push docker image into repository (https://hub.docker.com/)
  * dist.opt.sh :
    * for optimizer (also contains SALT simulator)
  * dist.salt.sh
    * only for SALT simulator

<hr>

### troubleshooting
* When visualization Server can not get realtime progress messages
  * set some parameters such as host, port, interval in simulation scenario file 
  * you can use virtual visualization server to check docker image is correctly packaged
    * location of virtual visualization server : traffic-simulator/test/libsalt/build/VisServer.out
    * to get VisServer.out, launch traffic-simulator/test/libsalt/c.sh

* If print **Syntax error: end of file unexpected (expecting "then")** when running the dist.opt.sh
  * try running this command
  ```shell script
  sed -i 's/\r$//' dist.opt.sh
  ```
