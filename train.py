from argparse import ArgumentParser
from omegaconf import OmegaConf

from HYPIR.trainer.sd2 import SD2Trainer


parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
config = OmegaConf.load(args.config)
if config.base_model_type == "sd2":
    trainer = SD2Trainer(config)
    trainer.run()
else:
    raise ValueError(f"Unsupported model type: {config.base_model_type}")
