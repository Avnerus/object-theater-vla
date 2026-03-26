"""
Global configuration and device management for Object Theater VLA.

This module defines the global DEVICE variable and dataclass-based
configuration management for hyperparameters.
"""

import torch

# Global device (cuda if available, else cpu)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
