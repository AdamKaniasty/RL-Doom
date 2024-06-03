import vizdoom as vzd
import os
from argparse import ArgumentParser
from multiprocessing import Process
from src.game.play import play
from src.game.train import train
from src.models.CustomPPO import CustomPPO_Model
from src.models.stable_baselines3_A2C import A2C_Model
import argparse

# CONFIG_PATH = os.path.join(vzd.scenarios_path, "basic.cfg")
CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Custom PPO model.')
    parser.add_argument('--epochs', type=int, default=8000, help='Number of epochs to train the model')

    args = parser.parse_args()

    model = CustomPPO_Model(
        CONFIG_PATH,
        mode='train'
    )

    model.train(args.epochs)
    model.save('cPPO_corridor_{}.zip'.format(args.epochs))

    # model = CustomPPO_Model(
    #     CONFIG_PATH,
    #     mode='test',
    #     pretrained='../models/weights/WEIGHTS_PPO_2_EXTENDED_100000x2'
    # )
    #
    # model.test()
