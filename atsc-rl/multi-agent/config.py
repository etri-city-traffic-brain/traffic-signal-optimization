TRAIN_CONFIG = {
    # 'libsalt_dir': '/home/pi/traffic-simulator/tools/libsalt',
    'IS_DOCKERIZE': True,
    'libsalt_dir': '/uniq/simulator/salt/tools/libsalt',
    'lr': 0.01,
    'lr_update_period': 5,
    'lr_update_decay': 0.9,
    'batch_size': 32,
    'replay_size': 2000,
    'network_size': [512, 512]
}