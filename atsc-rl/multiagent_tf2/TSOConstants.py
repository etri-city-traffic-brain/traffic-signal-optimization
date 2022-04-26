# -*- coding: utf-8 -*-

'''
constants
'''


class _INTERVAL_:
    '''
    time interval to control distributed learning
    '''
    # checking interval if learning is done
    LEARNING_DONE_CHECK = 2

    # wait time until next trial(attenpt)
    NEXT_TRIAL = 2




class _FN_PREFIX_ :
    '''
    file name prefix
    '''
    # (a prefix of) file name  to save optimal model info
    OPT_MODEL_INFO = "zz.optimal_model_info"

    # (a prefix of) file name to save a comparison of result
    RESULT_COMP = "zz.result_comp"



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



class _IMPROVEMENT_COMP_:
    '''
    used to indicate the improvement rate from DataFrame object
    '''
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
    TARGET_TLS = "targetTLs"
    INFER_TLS = "inferTLs"
    INFER_MODEL_NUMBER = "infer_model_number"
    CTRL_DAEMON_ARGS = "ctrl_daemon_args"

