import vizdoom as vzd
import os
import argparse

from src.models.CustomPPO import CustomPPO_Model

# CONFIG_PATH = os.path.join(vzd.scenarios_path, "basic.cfg")
CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Custom PPO model.')
    parser.add_argument('--epochs', type=int, default=8000, help='Number of epochs to train the model')

    args = parser.parse_args()
    iter = 2
    model = CustomPPO_Model(
        CONFIG_PATH,
        mode='train',
        pretrained=f'./src/models/weights/WEIGHTS_PPO_1_EXTENDEDPLUSREWARD_{iter}00000'

    )
    model.train(100000)
    model.save(f'WEIGHTS_PPO_1_EXTENDEDPLUSREWARD_{iter + 1}00000')

    # model = CustomPPO_Model(
    #     CONFIG_PATH,
    #     mode='test',
    #     pretrained='../models/weights/WEIGHTS_PPO_2_EXTENDED_100000x2'
    # )
    #
    # model.test()
