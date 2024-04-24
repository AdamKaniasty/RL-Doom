import vizdoom as vzd
import os
from src.models.stable_baselines3_A2C import A2C_Model

os.chdir("../..")

# CONFIG_PATH = os.path.join(vzd.scenarios_path, "basic.cfg")
CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

if __name__ == '__main__':
    model = A2C_Model(
        CONFIG_PATH,
        mode='train'
    )
    # 5K -> around 8:30 minutes
    model.train(5000)
    model.save('a2c_vizdoom_5K.zip')
