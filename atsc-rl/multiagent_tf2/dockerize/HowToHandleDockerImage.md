

### docker related commands

* docker file creation  
    ```
    [%]ls
    Dockerfile
  
    [%] cat Dockerfile 
    FROM ubuntu:18.04
    ENTRYPOINT ["echo", "hello... from my first docker image"]
    ENTRYPOINT ["sleep", "10"]
    ```


* image build
    ```
    [%] sudo docker build -t myimage:1.0 .
    ```


* execute
    ```
    [%] sudo docker run myimage:1.0
    [%] sudo docker run -it -v /home/hunsooni/z.tip/docker/uniq.dockerizing/to_install_uniq/optimizer/shared:/uniq/optimizer/shared my_uniq:0.2.a  run.salt.sh
        -v /path/to/our/workspace/foo:/uniq/foo
            /path/to/our/workspace/foo 를 이미지상의 /uniq/foo 로 메핑
    ```


* retrieve image  
    ```
    [%] sudo docker images -q
    ```


* stop image  
    ``` 
    [%] sudo docker stop [image_id]
    ```

* remove image
    ``` 
    [%] sudo docker rmi [image_id]
        sudo docker rmi -f [image_id]
        sudo docker rmi $(docker images -q) : remove all images
    ```

    
<hr>

### container handling 
ref. https://javacan.tistory.com/entry/docker-start-2-running-container

* show container name  
  docker container ls
    ``` 
    [%] sudo docker container ls
    ```

* attach terminal to running container  
sudo docker exec -it _container_name /bin/bash
    ```
    [%] sudo docker exec -it uniq_opt /bin/bash
    ```  


* start container  
docker start/restart _container_name_
  ``` 
  [%]  sudo docker start uniq_opt 
  [%]  sudo docker restart uniq_opt 
  ```


* pause/pause container  
docker pause/unpause _container_name_
  ```
  [%]  sudo docker pause uniq_opt 
  [%]  sudo docker unpause uniq_opt 
  ```


* stop container  
docker stop _container_name_
  ```
  [%]  sudo docker stop uniq_opt
  ```

* remove container  
docker rm _container_name_
  ```
  [%]  sudo docker rm uniq_opt 
  ```
	
