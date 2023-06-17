from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimDepthSensor
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)


@registry.register_sensor
class NewDepthSensor(HabitatSimDepthSensor):
    cls_uuid: str = "new_depth"
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "new_depth"
