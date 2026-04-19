from argparse import ArgumentParser
from omegaconf import OmegaConf

from HYPIR.trainer.sd2 import SD2Trainer
from HYPIR.trainer.sd2_alignment import SD2AlignmentTrainer
from HYPIR.trainer.sd2_alignment_stage1 import SD2AlignmentStage1Trainer


parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
config = OmegaConf.load(args.config)
if config.base_model_type == "sd2":
    trainer = SD2Trainer(config)
elif config.base_model_type == "sd2_alignment":
    trainer = SD2AlignmentTrainer(config)
elif config.base_model_type == "sd2_alignment_stage1":
    trainer = SD2AlignmentStage1Trainer(config)
else:
    raise ValueError(f"Unsupported model type: {config.base_model_type}")
trainer.run()
