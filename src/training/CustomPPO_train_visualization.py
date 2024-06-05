import vizdoom as vzd
import os
import argparse

from src.models.CustomPPO import CustomPPO_Model

# CONFIG_PATH = os.path.join(vzd.scenarios_path, "basic.cfg")
CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Custom PPO model.')
    parser.add_argument('--epochs', type=int, default=8192, help='Number of epochs to train the model')

    args = parser.parse_args()
    iter = 0
    model_name = 'WEIGHTS_PPO_2_FINALREWARD_' + str(args.epochs * iter)
    model = CustomPPO_Model(
        CONFIG_PATH,
        mode='train',
        # pretrained='./src/models/weights/' + model_name
    )
    model.train(args.epochs)
    # model.save(f'WEIGHTS_PPO_2_FINALREWARD_{args.epochs * (iter + 1)}')
