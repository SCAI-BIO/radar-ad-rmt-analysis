# functions
import random


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility"""

    import numpy as np

    # Conditionally import torch if available
    try:
        import torch

        has_torch = True
    except ImportError:
        has_torch = False

    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Scikit-learn (uses numpy's random state)
    from sklearn.utils import check_random_state

    check_random_state(seed)

    # If using PyTorch
    if has_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
