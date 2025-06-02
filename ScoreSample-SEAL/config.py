from dataclasses import dataclass, field
import torch
from model.score_gnn import (
    HadamardMLPPredictor,
    DotProductPredictor,
    ConcatMLPPredictor,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# train_data = torch.load(f"./data/{dataset}/split/train_data.pt").to(device)


@dataclass
class ScoreGNNConfig:
    gnn_type: str = "gcn"
    hidden_dim: int = 256
    output_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.5
    lr: float = 0.01
    epochs: int = 200
    predictor: object = HadamardMLPPredictor(input_dim=128).to(device)


@dataclass
class ScoreSampleSEALConfig:
    hidden_dim: int = 32
    num_layers: int = 3
    k: float = 0.6
    lr: float = 0.0001
    epochs: int = 50


@dataclass
class ScoreSamplerConfig:
    k_min: int = 60
    alpha: int = 40
    beta: int = 20
    gamma: int = 2
    num_hops: int = 3
    score_fn: str = "gnn"


@dataclass
class Config:
    version: str = "Github"
    dataset: str = "Github"
    seed: int = 2025
    data_init_num_features: int = (
        torch.load(f"./data/{dataset}/split/train_data.pt").to(device).num_features
    )
    device: torch.device = device
    scoregnn: ScoreGNNConfig = field(default_factory=ScoreGNNConfig)
    ssseal: ScoreSampleSEALConfig = field(default_factory=ScoreSampleSEALConfig)
    scoresampler: ScoreSamplerConfig = field(default_factory=ScoreSamplerConfig)
