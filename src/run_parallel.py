import vizdoom as vzd
import os
from argparse import ArgumentParser
from multiprocessing import Process
from game.play import play

CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        dest="config",
        default=CONFIG_PATH,
        nargs="?",
    )
    parser.add_argument(
        "-t",
        "--ticrates",
        default=[70],
        nargs="+",
    )
    args = parser.parse_args()

    processes = []
    for tickrate in args.ticrates:
        p = Process(target=play, args=(args.config, tickrate))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
