from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

#todo remove config.py.... get thw value of network_size & optimizer from command line
#
## add following func into TSOUtil.py(after str2bool())
# def stringToIntList(v):
#     import argparse
#     tokens = v.split(',')
#     ret_val = []
#     for i in range(len(tokens)):
#         ret_val.append(int(tokens[i]))
#     # raise argparse.ArgumentTypeError('comma-seperated string value expected.')
#     return ret_val
#
## add following 2 lines to addArgumentsToParser() at TSOUtil.py
# parser.add_argument('--network-size', type=stringToIntList, default=[1024, 512, 512, 512, 512], help='size of network in ML model')
# parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer for ML model')
#
## add following dictionary into ppoTF2.py
# __OPTIMIZERS_DIC__={ "adadelta" : Adadelta,
#                      "adagrad"  : Adagrad,
#                      "adam"     : Adam,
#                      "adamax"   : Adamax,
#                      "ftrl"     : Ftrl,
#                      "nadam"    : Nadam,
#                      "rmsprop"  : RMSprop,
#                      "sgd"      : SGD
#                     }
# use following stmt
# policy = __OPTIMIZERS_DIC__[config['policy']]

TRAIN_CONFIG = {
    # 'libsalt_dir': '/home/tsoexp/z.docker_test/traffic-simulator/tools/libsalt',
    'network_size': (1024, 512, 512, 512, 512),
    'optimizer' : Adam,

    # not used.....
    # 'lr': 0.01,
    # 'lr_update_period': 5,
    # 'lr_update_decay': 0.9,
    # 'batch_size': 32,
    # 'replay_size': 2000,
    # 'rnd_network_size': (256,128)
}