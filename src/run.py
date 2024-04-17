import vizdoom as vzd
import os
from argparse import ArgumentParser
from multiprocessing import Process
from game.play import play
from game.train import train

CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

play(CONFIG_PATH, 70)
# train(CONFIG_PATH)
