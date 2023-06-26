# -*- coding: utf-8 -*-

class DBG_OPTIONS :

    WITH_SOME_FUNC = True
    WITH_DBG_MSG = True

    ## functions : Options which are related to function
    if WITH_SOME_FUNC:
        ##-- done
        RunWithWaitForDebug = False      # wait for debug
        ResultCompareSkipWarmUp = False # skip warm-up-time to compare result
        RunWithDistributed = True  # find & store optimal model info  for distributed learning
        MaintainServerThreadState = False  # maintain a state of thread
        AddControlCycleIntoProblemVar = False # add control_cycle into problemVar or not;
                         # problemVar is used to construct the file name where the trained model is stored

        RichActionOutput = True # actions#oofse#duration_per_phase

        YJLEE = True  # only for YJLEE script
        #V20230605 = True # try to fix bugs which are related to multiple SA
        DoStateAugmentation = False # try to fix bugs which are related to state augmentation

        ##-- ing
        IngCompResult = False
        MergeAfterNormalize = True  # ref. __getState() at SappoEnv.py
                   # merge after normalize when we do collect info about a given environment
        DoNormalize = True   # ref. __getState() at SappoEnv.py
                   # whether do normalize or not when we gather state info. ; default is True

        NewModelUpdate = True # model update using only some of the experiences stored in replay memory

        ActorCriticModelVersion = 1 # 1 : simple model. 2 : optimized model with resnet, regularizer


    ## bebug message : Options which are related to debug message
    if WITH_DBG_MSG :
        PrintCtrlDaemon = True          # print messages which are related to operations of CtrlDaemon
        PrintExecDaemon = True          # print messages which are related to operations of CtrlDaemon
        PrintFindOptimalModel = False   # print messages which are related to optimal model finding
        PrintGeneratedCommand = True    # print messages which are related to command generation
        PrintImprovementRate = True     # print messages which are related to improvement rate
        PrintMsg = False                # print messages btn Controller Daemon and Execution Daemon
        PrintReplayMemory = True        # print messages which are related to replay-memory
        PrintResultCompare = False       # print messages which are related to result comparison
        PrintRewardMgmt = False         # print messages which are related to reward mgmt
        PrintSaRelatedInfo = False      # print SA related info
        PrintServingThread = True       # print messages which are related to serving thread
        PrintStep = False               # print progress msgs every step : inferred actions
        PrintTrain = True               # print messages which are related to train such as episode elapsed time, gc time, avg reward, ...
        PrintState = False              # print observateion
        PrintAction = False             # print action related things
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

        PrintState = True              # print observateion
        PrintAction = True             # print action related things



def waitForDebug(msg):
    if DBG_OPTIONS.RunWithWaitForDebug:
        return input("wait... {}\n\t\tenter if you want to keep going".format(msg))
    else:
        print(msg)
        return 0