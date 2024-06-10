from dataclasses import dataclass, field

__all__ = ["BaseConfig"]


@dataclass
class BaseConfig:
    name: str = field(
        default="BaseConfig",
        metadata={"help": "Name of the config"},
    )
