# -*- coding: utf-8 -*-

class DBG_OPTIONS :
    ## Options which are related maintaining states
    MaintainServerThreadState = True    # maintain a state of thread

    ## Options which are related to print message
    PrintCtrlDaemon = True          # print messages which are related to operations of CtrlDaemon
    PrintExecDaemon = True          # print messages which are related to operations of CtrlDaemon
    PrintFindOptimalModel = True    # print messages which are related to find optimal model
    PrintGeneratedCommand = True    # print messages which are related to generate command
    PrintImprovementRate = True     # print messages which are related to improvement rate
    PrintServingThread = True       # print messages which are related to serving thread

    PrintSaRelatedInfo = False      # print SA related info
    PrintStep = False               # print progress msgs every step : inferred actions

    PrintMsg = True                 # print messages

    ## Other Options
    RunWithWaitForDebug = False      # wait for debug
    # UseConfig = True    # use config dic instead of args
    # WithNewCode = True  # run with new code

    RunWithDistributed = True       # for distributed learning
    SimpleDistTestToSaveTestTime = False           # 시험 시간을 줄이기 위해 훈련 epoch를 1로 하며, 최적 모델을 항상 0으로 한다.


def waitForDebug(msg):
    if DBG_OPTIONS.RunWithWaitForDebug:
        return input("wait... {}\n\t\tenter if you want to keep going".format(msg))
    else:
        print(msg)
        return 0