from configs.config import config
from train.train import run


if __name__ == '__main__':
    print(config.general.experiment_name)
    run(config)
