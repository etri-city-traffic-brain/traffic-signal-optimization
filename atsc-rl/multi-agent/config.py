TRAIN_CONFIG = {
    'IS_DOCKERIZE': False,
    'libsalt_dir': '/home/pi/traffic-simulator/tools/libsalt',
    # 'libsalt_dir': '/uniq/simulator/salt/tools/libsalt',
    'lr': 0.01,
    'lr_update_period': 5,
    'lr_update_decay': 0.9,
    'batch_size': 32,
    'replay_size': 2000,
    # 'network_size': [16,32,64,128,64,32,16] # for keep or change
    'network_size': (512,512,512,512), # for keep or change
    'rnd_network_size': (512,512,512,512) # for keep or change
}