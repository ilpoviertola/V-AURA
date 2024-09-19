"""
Main entry point for training and testing.

Example:
python main.py
    config="./configs/experiments/vggsound/avclip/9cb-viscond-avclip-channel_concat-llama_like-ib_03.yaml" \
    model.sampler_config.params.num_layers=2 \
    trainer.logdir="./logs" \
    dataloader.data_dir="$VIDS_PATH" \
    dataloader.batch_size=10
"""

import os
import random
import logging
from typing import Dict

from omegaconf import OmegaConf
import torch
import numpy as np

from utils.train_utils import (
    get_curr_time_w_random_shift,
    update_cfg_with_ranks,
    is_master,
    set_logging_lvl_to,
)
from scripts.train import train
from scripts.generate import generate
from scripts.test import test


DEFAULT_CONFIG_PATH = "./configs/vaura_defaults.yaml"
torch.set_float32_matmul_precision("medium")


def set_env_variables():
    # checks if not run with torchrun or torch.launch.distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return
    # otherwise checks if on slurm cluster
    elif "SLURM_JOB_ID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NPROCS"]


def get_config(default_config_path: str = DEFAULT_CONFIG_PATH):
    args = OmegaConf.from_cli()
    cfg = OmegaConf.load(args.pop("config"))
    if cfg.action == "train":
        default_cfg = OmegaConf.load(default_config_path)
        cfg = OmegaConf.merge(default_cfg, cfg)

    # merge all but module args
    if "model" in args:
        module_args: Dict[str, Dict] = {"model": {}}
        for cfg_type in [
            "audio_encoder_config",
            "sampler_config",
            "feature_extractor_config",
            "pattern_provider_config",
        ]:
            module_args["model"][cfg_type] = args["model"].pop(cfg_type, {})
    else:
        module_args = {}
    cfg = OmegaConf.merge(cfg, args)

    # resolve modules etc.
    if "start_time" not in cfg or cfg.start_time is None:
        cfg.start_time = get_curr_time_w_random_shift()
    OmegaConf.register_new_resolver(
        "from_file", lambda rel_path: OmegaConf.load(rel_path)
    )
    OmegaConf.register_new_resolver("negation", lambda b: not bool(b))
    OmegaConf.resolve(cfg)

    # merge module args
    cfg = OmegaConf.merge(cfg, module_args)
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_logging_lvl_to(logging.WARNING)
    set_env_variables()
    cfg = get_config()
    update_cfg_with_ranks(cfg)

    if is_master(cfg):
        print("ACTION:", cfg.action)
        print("START TIME:", cfg.start_time)
        print("WORLD SIZE:", cfg.trainer.world_size)
        print("CONFIG:")
        print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.get("trainer", {}).get("seed", 666))

    if cfg.action == "train":
        train(cfg)
    elif cfg.action == "test":
        test(cfg)
    elif cfg.action == "eval":
        if is_master(cfg):
            print("use https://github.com/ilpoviertola/eval_generative_v2a_models")
    elif cfg.action == "generate":
        generate(cfg)
    elif cfg.action == "finetune":
        raise NotImplementedError
    else:
        raise NotImplementedError(f"Unknown action: {cfg.action}")


if __name__ == "__main__":
    main()
