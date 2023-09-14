from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from habitat.config.default_structured_configs import (
    SimulatorSensorConfig,
    SimulatorCameraSensorConfig,
)
import extensions.new_sensors
import extensions.new_environments


cs = ConfigStore.instance()


@dataclass
class DepthConfig:
    """Depth encoder config"""

    model_name: str = "openai/clip-vit-base-patch32"
    feature_dim: int = 768
    trainable: bool = False
    is_blind: bool = False
    pretrained: bool = True
    pretrained_weights: str = "data/checkpoints/TAC/best.pth"
    dropout: float = 0.0
    post_process: str = "normalize"  # for rearrange, use laryernorm


@dataclass
class NewDepthSensorConfig(SimulatorCameraSensorConfig):
    type: str = "NewDepthSensor"
    min_depth: float = 0.0
    max_depth: float = 10.0
    normalize_depth: bool = True
    uuid: str = "new_depth"


cs.store(group="habitat_baselines/rl/policy/depth", name="depth_base", node=DepthConfig)
cs.store(group="habitat_baselines/il/depth", name="depth_base", node=DepthConfig)
cs.store(
    group="habitat/simulator/agents/main_agent/sim_sensors/new_depth_sensor",
    name="new_depth_sensor",
    node=NewDepthSensorConfig,
)
