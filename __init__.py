from .node import *
from .install import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "AutomaticMask(segment anything)": AutomaticMask
}

__all__ = ['NODE_CLASS_MAPPINGS']


