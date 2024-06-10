from dataclasses import dataclass, field

from SSL_Interface.configs.baseConfig import BaseConfig

__all__ = [
    "BaseInterfaceConfig",
    "WeightedSumInterfaceConfig",
    "HierarchicalConvInterfaceConfig",
    "SingleLayerInterfaceConfig",
]


@dataclass
class BaseInterfaceConfig(BaseConfig):
    name: str = "BaseModelConfig"
    upstream_layer_num: int = 13
    upstream_feat_dim: int = 768
    normalize: bool = False


@dataclass
class SingleLayerInterfaceConfig(BaseInterfaceConfig):
    name: str = "SingleLayerInterfaceConfig"
    selected_layer: int = -1


@dataclass
class WeightedSumInterfaceConfig(BaseInterfaceConfig):
    name: str = "WeightedSumInterfaceConfig"


@dataclass
class HierarchicalConvInterfaceConfig(BaseInterfaceConfig):
    name: str = "HierarchicalConvInterfaceConfig"
    channel_change_strategy: str = "same_upstream"
    conv_kernel_size: int = 5
    conv_kernel_stride: int = 3
    output_dim: int = None
