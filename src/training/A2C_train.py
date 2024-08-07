import vizdoom as vzd
import os
from argparse import ArgumentParser
from multiprocessing import Process
from src.game.play import play
from src.game.train import train
from src.models.stable_baselines3_A2C import A2C_Model
import argparse

# CONFIG_PATH = os.path.join(vzd.scenarios_path, "basic.cfg")
CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

if __name__ == '__main__':
    # Training the model

    parser = argparse.ArgumentParser(description='Train the A2C model.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train the model')

    args = parser.parse_args()

    model = A2C_Model(
        CONFIG_PATH,
        mode='train'
    )
    # 5K -> around 8:30 minutes
    model.train(args.epochs)
    model.save('a2c_corridor_{}.zip'.format(args.epochs))
