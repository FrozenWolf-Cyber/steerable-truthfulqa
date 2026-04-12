from .base import Steer, VecSteer
from .caa import CAA
from .iti import ITI
from .repe import RepE
from .lin_act import LinAcT
from .mimic import MiMiC
from .pace import PaCESteerer

_REGISTRY = {
    "CAA": CAA,
    "ITI": ITI,
    "RepE": RepE,
    "LinAcT": LinAcT,
    "MiMiC": MiMiC,
}


def get_steer_model(name: str, **kwargs):
    if name == "NoSteer" or name is None:
        return None
    if name == "PaCE":
        raise ValueError("PaCE requires special initialization — use PaCESteerer directly.")
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown steer method: {name}. Available: {list(_REGISTRY)}")
    return cls(**kwargs)
