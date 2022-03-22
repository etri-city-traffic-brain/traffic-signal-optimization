TRAIN_CONFIG = {
    'IS_DOCKERIZE': False,
    'libsalt_dir': '/home/pi/traffic-simulator/tools/libsalt',
    # 'libsalt_dir': '/uniq/simulator/salt/tools/libsalt',
    'lr': 0.01,
    'lr_update_period': 5,
    'lr_update_decay': 0.9,
    'batch_size': 32,
    'replay_size': 2000,
    # 'network_size': (256,256,256,256,256,256,256,256,256,256,256,256), # for keep or change
    # 'network_size': (512,512,512,512,512), # for keep or change
    'network_size': (256,512,512,512,1024,1024), # for keep or change
    'rnd_network_size': (256,128) # for keep or change
}