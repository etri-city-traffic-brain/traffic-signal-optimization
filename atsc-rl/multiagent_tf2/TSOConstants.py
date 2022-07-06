# -*- coding: utf-8 -*-

'''
constants
'''


ONE_HOUR = 3600 # 1 hour is 3600 seconds(=60 * 60)
TEN_MINUTE = 600 # ten minute is 600 seconds(=10 * 60)
_RESULT_COMPARE_SKIP_ = TEN_MINUTE # ONE_HOUR

class _INTERVAL_:
    '''
    time interval to control distributed learning
    '''
    # checking interval if learning is done
    LEARNING_DONE_CHECK = 2

    # wait time until next trial(attenpt)
    NEXT_TRIAL = 2



#todo  change _FN_PREFIX_ --> _FN_, OPT_MODEL_INFO --> PREFIX_OPT_MODEL_INFO, RESULT_COMP -->PREFIX_RESULT_COMP
class _FN_PREFIX_ :
    '''
    file name prefix
    '''
    # (a prefix of) file name  to save optimal model info
    OPT_MODEL_INFO = "zz.optimal_model_info"

    # (a prefix of) file name to save a comparison of result
    RESULT_COMP = "zz.result_comp"

    # file name to save the history of distributed learning
    DIST_LEARNING_HISTORY = 'zz.dist_learning_history.csv'

    # (a prefix of) file name to save the contents of replay memory
    REPLAY_MEMORY = 'zz.replay_memory'



class _CHECK_ :
    '''
    const to save validation result
    '''
    NOT_READY = 0x0000
    ON_GOING  = 0x0001
    FAIL      = 0x0002
    SUCCESS   = 0x0004



class _STATE_:
    '''
    constants which indicate the state of daemon
    '''
    NOT_READY = 0x0000
    SEND_LOCAL_LEARNING_REQUEST = 0x0000
    RECV_LOCAL_LEARNING_DONE = 0x0000
    SEND_TERMINATION_REQUEST = 0x0000
    RECV_TERMINATION_DONE = 0x0000




class _MODE_ :
    '''
    mode
    '''
    TEST = "test"
    SIMULATE = "simulate"
    TRAIN = "train"



class _RESULT_COMP_:
    '''
    used to indicate the improvement rate from DataFrame object
    '''
    SIMULATION_OUTPUT='_PeriodicOutput.csv'
    PHASE_REWARD_OUTPUT='rl_phase_reward_output.txt'
    SPEED_GATHER_INTERVAL = 30 # seconds; interval for gathering the average speed of intersection
    ROW_NAME = "total"
    COLUMN_NAME = 'imp_SumTravelTime_sum_0hop'




class _MSG_TYPE_ :
    '''
    message types
    '''
    CONNECT_OK               = 0x0001   # connection is done
    LOCAL_LEARNING_REQUEST   = 0x0002   # do local learning
    LOCAL_RELEARNING_REQUEST = 0x0004   # do local re-learning
    LOCAL_LEARNING_DONE      = 0x0008   # local learning done
    TERMINATE_REQUEST        = 0x0010
    TERMINATE_DONE           = 0x0020
    READY                    = 0xFFFF   # ready for test



class _MSG_CONTENT_:
    '''
    element name of dictionary which construct msg_contents
    '''
    TARGET_TL = "targetTL"
    INFER_TL = "inferTL"
    INFER_MODEL_NUMBER = "infer_model_number"
    CTRL_DAEMON_ARGS = "ctrl_daemon_args"


class _REWARD_GATHER_UNIT_:
    '''
    reward gathering unit 보상을 어떤 단위로 수집할 것인가... 교차로, 교차로 그룹, ...
    '''
    TL = 'TL'    # gather only target & gathering unit is TL
    SA = 'SA'    # gather only target & gathering unit is SA
    ENV = 'ENV'  # gather whole traffic env & gathering unit is env