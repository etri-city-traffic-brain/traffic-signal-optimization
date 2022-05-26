# -*- coding: utf-8 -*-

class DBG_OPTIONS :

    WITH_DBG_MSG = False #

    RunWithDistributed = True  # find & store optimal model info  for distributed learning
    PrintTrain = True  # print messages which are related to train


    CARE_CUMULATIVE = True # care cumulative learning.... todo should delete after test

    USE_EXPLORATION_EPSILON = False # use epsilon for exploration.... todo should delete after test

    KEEP_OPTIMAL_MODEL_NUM = True # to store(keep) all optimal model num when we do experiment ... todo should delete after test

    if WITH_DBG_MSG :
        ## Options which are related maintaining states
        MaintainServerThreadState = True    # maintain a state of thread

        ## Options which are related to print message
        PrintCtrlDaemon = True          # print messages which are related to operations of CtrlDaemon
        PrintExecDaemon = True          # print messages which are related to operations of CtrlDaemon
        PrintFindOptimalModel = False    # print messages which are related to find optimal model
        PrintGeneratedCommand = True    # print messages which are related to generate command
        PrintImprovementRate = True     # print messages which are related to improvement rate
        PrintMsg = False                 # print messages
        PrintReplayMemory = True              # print messages which are related to replay memory
        PrintResultCompare = True       # print messages which are related to result comparison
        PrintRewardMgmt = False          # print messages which are related to reward mgmt
        PrintSaRelatedInfo = False      # print SA related info
        PrintServingThread = True       # print messages which are related to serving thread
        PrintStep = False               # print progress msgs every step : inferred actions
        # PrintTrain = True               # print messages which are related to train


        ## Other Options
        RunWithWaitForDebug = True      # wait for debug
        # UseConfig = True    # use config dic instead of args
        # WithNewCode = True  # run with new code

    else:
        ## Options which are related maintaining states
        MaintainServerThreadState = False  # maintain a state of thread

        ## Options which are related to print message
        PrintCtrlDaemon = False  # print messages which are related to operations of CtrlDaemon
        PrintExecDaemon = False  # print messages which are related to operations of CtrlDaemon
        PrintFindOptimalModel = False  # print messages which are related to find optimal model
        PrintGeneratedCommand = False  # print messages which are related to generate command
        PrintImprovementRate = False  # print messages which are related to improvement rate
        PrintMsg = False  # print messages
        PrintReplayMemory = False  # print messages which are related to replay memory
        PrintResultCompare = False  # print messages which are related to result comparison
        PrintRewardMgmt = False  # print messages which are related to reward mgmt
        PrintSaRelatedInfo = False  # print SA related info
        PrintServingThread = False  # print messages which are related to serving thread
        PrintStep = False  # print progress msgs every step : inferred actions
        # PrintTrain = True  # print messages which are related to train

        ## Other Options
        RunWithWaitForDebug = False  # wait for debug



def waitForDebug(msg):
    if DBG_OPTIONS.RunWithWaitForDebug:
        return input("wait... {}\n\t\tenter if you want to keep going".format(msg))
    else:
        print(msg)
        return 0