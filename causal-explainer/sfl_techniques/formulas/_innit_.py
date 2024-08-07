# __init__.py
import torch
from .ochiai import calculate_ochiai
from .tarantula import calculate_tarantula
from .zoltar import calculate_zoltar
from .wong1 import calculate_wong1
from .fault_localization_metrics import FaultLocalizationMetrics

# You can also define __all__ to control what gets imported with "from package import *"
__all__ = ['calculate_ochiai', 'calculate_tarantula', 'calculate_zoltar', 'calculate_wong1', 'FaultLocalizationMetrics']
