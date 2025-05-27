from dataclasses import dataclass, field
import torch
from model.score_gnn import (
    HadamardMLPPredictor,
    DotProductPredictor,
    ConcatMLPPredictor,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = torch.load("./data/Cora/split/train_data.pt").to(device)


@dataclass
class ScoreGNNConfig:
    hidden_dim: int = 256
    output_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.5
    lr: float = 0.01
    epochs: int = 200


@dataclass
class ScoreSampleSEALConfig:
    hidden_dim: int = 32
    num_layers: int = 3
    k: float = 0.6
    lr: float = 0.0001
    epochs: int = 50


@dataclass
class Config:
    drnl: bool = True
    seed: int = 2025
    data_init_num_features: int = train_data.num_features
    device: torch.device = device
    scoregnn: ScoreGNNConfig = field(default_factory=ScoreGNNConfig)
    predictor: object = HadamardMLPPredictor(input_dim=128).to(device)
    k_top: int = 80
    num_hops: int = 3
    ssseal: ScoreSampleSEALConfig = field(default_factory=ScoreSampleSEALConfig)
