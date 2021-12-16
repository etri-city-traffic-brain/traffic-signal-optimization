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
│     ├── opt_data  : used to dockerize optimizer
│     │     ├── magic  
│     │     │     ├── doan(without dan).tss.xml  
│     │     │     └── doan_20210401.edg.xml  
│     │     └── scenario  
│     │           └── doan  
│     │                 ├── dj_doan_kaist_2h.rou.xml
│     │                 ├── doan(without dan).tss.xml
│     │                 ├── doan.con.xml
│     │                 ├── doan.edg.xml
│     │                 ├── doan.nod.xml
│     │                 ├── doan.tss.xml
│     │                 ├── doan_2021.scenario.json
│     │                 ├── doan_20210401.edg.xml
│     │                 ├── doan_20210421.tss.xml
│     │                 ├── doan_2021_ft.scenario.json
│     │                 └── doan_2021_test.scenario.json
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
└── boost_1_69_0.tar.bz2     : bootstrap library

```
