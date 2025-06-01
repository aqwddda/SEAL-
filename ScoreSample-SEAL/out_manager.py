import os
import torch
import json
import logging
from datetime import datetime


def get_out_dir(config, base_dir="./out"):
    """
    根据 config 生成唯一输出目录（如 k20_hop5、k20_hop5_1 等）
    """
    dataset = getattr(config, "dataset", "unknown")
    k_min = getattr(config.scoresampler, "k_min", "unknown")
    num_hops = getattr(config.scoresampler, "num_hops", "unknown")
    version = getattr(config, "version", "unknown")
    base_folder_name = f"{dataset}_k{k_min}_hop{num_hops}_{version}"
    out_dir = os.path.join(base_dir, base_folder_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        return out_dir

    # 如果已存在，则自动编号
    idx = 1
    while True:
        new_out_dir = f"{out_dir}_{idx}"
        if not os.path.exists(new_out_dir):
            os.makedirs(new_out_dir)
            return new_out_dir
        idx += 1


def get_existing_out_dir(config, base_dir="./out"):
    dataset = getattr(config, "dataset", "unknown")
    k_min = getattr(config.scoresampler, "k_min", "unknown")
    num_hops = getattr(config.scoresampler, "num_hops", "unknown")
    version = getattr(config, "version", "unknown")
    base_folder_name = f"{dataset}_k{k_min}_hop{num_hops}_{version}"

    # 查找所有匹配的目录
    matching_dirs = []
    for dir_name in os.listdir(base_dir):
        if dir_name == base_folder_name:
            matching_dirs.append((0, dir_name))
        elif dir_name.startswith(base_folder_name + "_"):
            try:
                idx = int(dir_name.split("_")[-1])
                matching_dirs.append((idx, dir_name))
            except ValueError:
                continue

    if not matching_dirs:
        raise FileNotFoundError(f"No folder found for config k{k_min}_hop{num_hops}")

    # 返回编号最大的目录
    max_idx_dir = max(matching_dirs, key=lambda x: x[0])[1]
    return os.path.join(base_dir, max_idx_dir)


def save_config(config, save_dir):

    config_path = os.path.join(save_dir, "config.json")

    def convert(obj):
        if isinstance(obj, torch.device):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        if obj.__class__.__name__ in {
            "HadamardMLPPredictor",
            "DotProductPredictor",
            "ConcatMLPPredictor",
        }:
            return obj.__class__.__name__
        if hasattr(obj, "__dict__"):
            return {k: convert(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(config_path, "w") as f:
        json.dump(convert(config), f, indent=4)
    print(f"Configuration saved to: {config_path}")


def setup_logging(log_path):
    """
    设置 logging 日志，输出到文件 + 控制台。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )
