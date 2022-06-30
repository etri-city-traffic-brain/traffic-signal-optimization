# -*- coding: utf-8 -*-

class DBG_OPTIONS :

    WITH_SOME_FUNC = True
    WITH_DBG_MSG = False

    ## functions : Options which are related to function
    if WITH_SOME_FUNC:
        ##-- done
        RunWithWaitForDebug = False      # wait for debug
        ResultCompareSkipWarmUp = False # skip warm-up-time to compare result
        RunWithDistributed = True  # find & store optimal model info  for distributed learning
        MaintainServerThreadState = False  # maintain a state of thread


        ##-- ing
        IngCompResult = False
        NEW_PPO = True # std 값 변경 되게 하기 위함  #  현재 적용되어 있지 않음. diff ppoTF2.py.org ppoTF2v2.py



    ## bebug message : Options which are related to debug message
    if WITH_DBG_MSG :
        PrintCtrlDaemon = True          # print messages which are related to operations of CtrlDaemon
        PrintExecDaemon = True          # print messages which are related to operations of CtrlDaemon
        PrintFindOptimalModel = False   # print messages which are related to optimal model finding
        PrintGeneratedCommand = True    # print messages which are related to command generation
        PrintImprovementRate = True     # print messages which are related to improvement rate
        PrintMsg = False                # print messages btn Controller Daemon and Execution Daemon
        PrintReplayMemory = True        # print messages which are related to replay-memory
        PrintResultCompare = True       # print messages which are related to result comparison
        PrintRewardMgmt = False         # print messages which are related to reward mgmt
        PrintSaRelatedInfo = False      # print SA related info
        PrintServingThread = True       # print messages which are related to serving thread
        PrintStep = False               # print progress msgs every step : inferred actions
        PrintTrain = True               # print messages which are related to train such as episode elapsed time, gc time, avg reward, ...

    else:
        PrintCtrlDaemon = False        # print messages which are related to operations of CtrlDaemon
        PrintExecDaemon = False        # print messages which are related to operations of CtrlDaemon
        PrintFindOptimalModel = False  # print messages which are related to optimal model finding
        PrintGeneratedCommand = False  # print messages which are related to command generation
        PrintImprovementRate = False   # print messages which are related to improvement rate
        PrintMsg = False               # print messages btn Controller Daemon and Execution Daemon
        PrintReplayMemory = False      # print messages which are related to replay-memory
        PrintResultCompare = False     # print messages which are related to result comparison
        PrintRewardMgmt = False        # print messages which are related to reward mgmt
        PrintSaRelatedInfo = False     # print SA related info
        PrintServingThread = True      # print messages which are related to serving thread
        PrintStep = False              # print progress msgs every step : inferred actions
        PrintTrain = True              # print messages which are related to train such as episode elapsed time, gc time, avg reward, ...



def waitForDebug(msg):
    if DBG_OPTIONS.RunWithWaitForDebug:
        return input("wait... {}\n\t\tenter if you want to keep going".format(msg))
    else:
        print(msg)
        return 0