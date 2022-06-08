### should check 
* version of binaries : simulator, optimizer
  * binaries will be updated  before making a docker image if you use a distribution script(dist.*.sh)
  
  
<hr>

### files in 'to_install_uniq' Directory
```
.  
├── README.md  
│ 
├── additional  : addition files
│     │      
│     └── salt_data  : used to dockerize salt
│           ├── salt.py
│           ├── salt.sh
│           └── sample 
│               ├── connection.xml
│               ├── edge.xml
│               ├── node.xml
│               ├── route.xml
│               ├── sample.json
│               └── tss.xml
│ 
├── boost_1_69_0.tar.bz2     : bootstrap library
│
└── boost_1_71_0.tar.bz2     : bootstrap library

```
