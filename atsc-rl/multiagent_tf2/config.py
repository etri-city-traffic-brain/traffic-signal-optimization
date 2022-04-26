from tensorflow.keras.optimizers import Adam

TRAIN_CONFIG = {
    'libsalt_dir': '/home/tsoexp/z.docker_test/traffic-simulator/tools/libsalt',
    'network_size': (1024, 512, 256, 128, 64),
    'optimizer' : Adam,

    # 'libsalt_dir': '/home/pi/traffic-simulator/tools/libsalt',
    # 'network_size': (512, 512, 512, 512),


    # not used.....
    # 'lr': 0.01,
    # 'lr_update_period': 5,
    # 'lr_update_decay': 0.9,
    # 'batch_size': 32,
    # 'replay_size': 2000,
    # 'rnd_network_size': (256,128)
}