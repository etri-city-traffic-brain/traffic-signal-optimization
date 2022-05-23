
## Troubleshooting
* do remove simulator msg 
* how to increase the # of max open file
* Python comment : Python Integrated Tools
* when class member function __funcFoo() is not called
* when python process terminated with "killed" message
* when you meet the "ModuleNotFoundError: No module named "libsalt" "
* when you meet the sudden disconnection from the SSH session with the message: “client_loop: send disconnect: Broken pipe”
* When tensorboard does not work
* when the simulation/learning is too slow
* when training time becomes longer

<hr>


### do remove simulator mgs 
* comment couts
  * traffic-simulator/src/Object/TrafficSignal/TrafficSignal.cpp

### Increase the number of max open file
* /etc/security/limits.conf 에 다음 내용 추가
    ```
    * hard nofile 1024000
    * soft nofile 1024000
    tsoexp hard nofile 1024000
    tsoexp soft nofile 1024000
    ```
* /etc/sysctl.conf에 다음 내용 추가
    ```
    # Uncomment the following to stop low-level messages on console
    #kernel.printk = 3 4 1 3
    fs.file-max = 2097152  #<--- 추가 
    ```
* .bashrc에 다음 내용 추가
   ulimit -n 1024000
* ref
  * https://blog.hbsmith.io/too-many-open-files-%EC%97%90%EB%9F%AC-%EB%8C%80%EC%9D%91%EB%B2%95-9b388aea4d4e
  * http://www.linuxdata.org/bbs/board.php?bo_table=OS&wr_id=33

### python comment : Python Integrated Tools
* File | Settings | Tools | Python Integrated Tools
   ```
   Docstrings --> Docstring format
       reStructuredText
         '''
         :param a:
         :param b:
         :return:
         '''
  
       EpyText
         '''
         @param self:
         @param myParam1:
         @param myParam2:
         @return:
        ```
  
### when class member function __funcFoo() is not called
* be careful when you use double underscore as a start of func name  
  * can not be called outside of defined class 
    * if name of member function is start with double underscode(i.e., __funcFoo )

  

### when python process terminated with "killed" message
* reason :  out of memory
* check & confirm :  dmesg | grep -E -i -B100 'killed process' 
* solve
  * use garbage collector
    * ref
      * ref. https://yjs-program.tistory.com/10    
      * https://twinparadox.tistory.com/623  
      * https://medium.com/dmsfordsm/garbage-collection-in-python-777916fd3189
    * sample
       ```python
       import gc
       collected = gc.collect()
       print("collected={}".format(collected))
       ```
    * disadvantage : can be slower
  * use del obj
* consider python magic method
  * __delete__ / __del__ / __delattr__ / __delitem__


### when you meet the "ModuleNotFoundError: No module named "libsalt" "
* launch with path of SALT_HOME
  ```shell
  [%] SALT_HOME=/home/tsoexp/z.docker_test/traffic-simulator run.py --mode simulate ....
  ```

### when you meet the sudden disconnection from the SSH session with the message: “client_loop: send disconnect: Broken pipe”
* client_loop: send disconnect: Broken pipe
* add following to the SSH configuration 
  * .ssh/config for just you (or /etc/ssh/ssh_config for all user)
  ```text
  Host *
  ServerAliveInterval 120
  TCPKeepAlive no
  ```
  * directive “ServerAliveInterval“ will send a null packet to the server at 120-second intervals to keep the session alive and thus prevent it from closing abruptly.
  * directive "TCPKeepAlive" specifies whether the system should send TCP keepalive messages to the other side
* You can also make your OpenSSH server keep alive all connections with clients 
  * by adding the following to /etc/ssh/sshd_config:
    ```text
    ClientAliveInterval 300
    ClientAliveCountMax 2
    ```
* ref
  * https://en.stealthsettings.com/fix-ssh-error-terminal-linux-client_loop-send-disconnect-broken-pipe.html
  * http://egloos.zum.com/mcchae/v/11382363
  * https://patrickmn.com/aside/how-to-keep-alive-ssh-sessions/


### When tensorbord does not work

    ```python
    tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
    tf.compat.v1.disable_eager_execution()  # usually using this for fastest performance
              # if this code for fast performance is executed, tensorboard does not work
    ``` 

### when the simulation/learning is too slow
* check the version of python : python version should be equal
  * Python version used to create the libsalt library
  * Python version used to execute the application
* recommended that you should make the libsalt library using the environment for the reinforcement learning program execution
  * After creating the reinforcement learning program execution environment

### when training time becomes longer
* training time become longer multiple times when we use config.py 
  * more then 5 ~ 10 times longer 
  * example :
     ```
     sim_period = TRAIN_CONFIG['sim_period] 
     ```
* use argument passing instead of configuration file
  * sim_period = args.sim_period