import vizdoom as vzd
import os
from argparse import ArgumentParser
from multiprocessing import Process
from src.game.play import play
from src.game.train import train
from src.models.stable_baselines3_A2C import A2C_Model

# CONFIG_PATH = os.path.join(vzd.scenarios_path, "basic.cfg")
CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

# play(CONFIG_PATH, 35)


if __name__ == '__main__':
    # Loading and testing the pretrained model

    model = A2C_Model(
        CONFIG_PATH,
        mode='test',
        pretrained='./src/models/weights/a2c_corridor_10000.zip'
    )

    model.test()

