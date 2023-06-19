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
        V20230605_StateAugmentation = False # try to fix bugs which are related to state augmentation
            # @todo False는
                #--- run_off_ppo_single.py : SA 101, SA 104 2개 SA 성공, 주기가 다른 2 SA 최적화시 오류
                #--- run_off_ppo_single_mod.py : 도안지역  4개 SA 성공
                #--- run_dist.py : 도안지역  4개 SA 성공

                #####
                # 주기가 다른 경우에 정상 동작하는것 같으나 재현 메모리에 저장되는 상태의 모양이 다르다
                #---- 주기가 같으면 일정하게 (1,13)으로 저장된다.
                #---- SA107의 경우 (1,13) 모양으로만 저장되어야 하나 (1,13) (13,)의 두가지 모양으로 저장된다.
                    #14% done
                    ### [ 241500 , 241500 ] in Env::_reshape() 0 before reshape(1,-1)... state[0].shape=(13,)  idx_of_act_as=[0]
                    #### [ 241500 , 241500 ] in Env::_reshape() 0 after reshape(1,-1)... state[0].shape=(1, 13)  idx_of_act_as=[0]
                    #[ 241136 , 241136 ] DELETE_1 SA_107 in memory.store() type(state)=<class 'numpy.ndarray'> shape=(13,) len=13
                    #action sampling, train
                    #19% done
                    #### [ 241500 , 241500 ] in Env::_reshape() 0 before reshape(1,-1)... state[0].shape=(13,)  idx_of_act_as=[0, 1]
                    #### [ 241500 , 241500 ] in Env::_reshape() 0 after reshape(1,-1)... state[0].shape=(1, 13)  idx_of_act_as=[0, 1]
                    #### [ 241500 , 241500 ] in Env::_reshape() 1 before reshape(1,-1)... state[1].shape=(119,)  idx_of_act_as=[0, 1]
                    #### [ 241500 , 241500 ] in Env::_reshape() 1 after reshape(1,-1)... state[1].shape=(1, 119)  idx_of_act_as=[0, 1]
                    #[ 241136 , 241136 ] DELETE_1 SA_107 in memory.store() type(state)=<class 'numpy.ndarray'> shape=(1, 13) len=1
                    #[ 241136 , 241136 ] DELETE_1 SA_101 in memory.store() type(state)=<class 'numpy.ndarray'> shape=(119,) len=119
                #---- 상태의 모양/값이 중간에 변경되는것 같다.
                #------- 주기가 같으면 일정하게 (1,13)으로 저장되는데... 혹시 쓰레드라서 문제가 되는 것일까?

        V20230605_PR_DBG_MSG=False
            ##
            ## 복사 후 관련 내용 정리 후 깃허브에 Commit/Push

            # @todo True 는 ?
                #--- run_off_ppo_single.py : SA 101, SA 104 2개 SA 성공
                #--- run_off_ppo_single_mod.py :
                #--- run_dist.py : 1개 성공,  SA 101, SA 107 2개 SA 오류
                #
            # @todo 모델 생성시 상태크기가 올바르게 설정되는지 확인하자.
                #--- Env::get_agent_configuration()에서   state_size = (state_space+self.step_size[i],) 계산한다.


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