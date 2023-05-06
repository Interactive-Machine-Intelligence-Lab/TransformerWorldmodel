import hydra
from omegaconf import DictConfig

from trainer import Trainer


def main():
    trainer = Trainer()
    trainer.run()


if __name__ == "__main__":
    main()
