# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import numpy as np


class BaseScheduler:
    def __init__(self):
        pass

    def schedule(self, n: int) -> float:
        pass

    def __call__(self, n: int) -> float:
        assert hasattr(self, 'start')
        
        return self.schedule(n) * self.start


class ExponentialDecayScheduler(BaseScheduler):
    def __init__(self, start: float, end: float, decay_every_step: int, scale_factor: float) -> None:
        super().__init__()
        self.decay_every_step = decay_every_step
        self.scale_factor = scale_factor

        self.start = start
        self.end = end
        self.current = start
        
    def schedule(self, n: int) -> float:
        if not n % self.decay_every_step:
            res = np.exp(-self.scale_factor*n) * self.start
            self.current = max(self.end, res)
            
        return self.current / self.start


class LambdaWarmUpCosineScheduler(BaseScheduler):
    def __init__(self, warm_up_steps: int, max_decay_steps: int, min_: float, max_: float, start: float) -> None:
        super().__init__()
        assert (max_decay_steps >= warm_up_steps)
        
        self.warm_up_steps = warm_up_steps
        self.start = start
        self.min_ = min_
        self.max_ = max_
        self.max_decay_steps = max_decay_steps
        self.last = 0.

    def schedule(self, n: int) -> float:
        if n < self.warm_up_steps:
            res = (self.max_ - self.start) / self.warm_up_steps * n + self.start
            self.last = res
        else:
            t = (n - self.warm_up_steps) / (self.max_decay_steps - self.warm_up_steps)
            t = min(t, 1.0)
            res = self.min_ + 0.5 * (self.max_ - self.min_) * (1 + np.cos(t * np.pi))
            self.last = res
    
        return res / self.start
    

class LambdaWarmUpLinearScheduler(BaseScheduler):
    def __init__(self, warm_up_steps: int, max_decay_steps: int, min_: float, max_: float, start: float) -> None:
        super().__init__()
        assert (max_decay_steps >= warm_up_steps)
        
        self.warm_up_steps = warm_up_steps
        self.start = start
        self.min_ = min_
        self.max_ = max_
        self.max_decay_steps = max_decay_steps
        self.last = 0.
        
    def schedule(self, n: int) -> float:
        if n < self.warm_up_steps:
            res = (self.max_ - self.start) / self.warm_up_steps * n + self.start
            self.last = res
        else:
            res = self.min_ + (self.max_ - self.min_) * (max_decay_steps - n) / max_decay_steps
            self.last = res
    
        return res / self.start
