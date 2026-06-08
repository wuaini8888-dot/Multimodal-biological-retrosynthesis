import torch
import numpy as np
import random


def set_seed(seed):
    """
    Sets the random seed for reproducibility across all random number generators.
    Ensures deterministic behavior for PyTorch operations.

    Args:
        seed (int): The integer value to use as the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


class NoamOpt:
    """
    Optim wrapper that implements learning rate schedule based on the paper
    "Attention Is All You Need".

    The learning rate increases linearly for the first `warmup` steps,
    and decreases proportionally to the inverse square root of the step number thereafter.
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        """
        Args:
            model_size (int): Hidden dimension size of the model (d_model).
            factor (float): Multiplicative factor for the learning rate.
            warmup (int): Number of warmup steps.
            optimizer (torch.optim.Optimizer): The underlying PyTorch optimizer.
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Updates the learning rate and performs an optimization step."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Calculates the learning rate for the current step."""
        if step is None:
            step = self._step

        return self.factor * (
                self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.optimizer.zero_grad()