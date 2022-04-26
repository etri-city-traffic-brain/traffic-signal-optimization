
## Troubleshooting
* Increase the # of max open file
* Python comment : Python Integrated Tools
* When tensorboard does not work
* when python process terminated with "killed" message

<hr>

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



### When tensorbord does not work

    ```python
    tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
    tf.compat.v1.disable_eager_execution()  # usually using this for fastest performance
              # if this code for fast performance is executed, tensorboard does not work
    ``` 
