# Copyright (c) OpenMMLab. All rights reserved.

import logging
import os
import copy
import time
import warnings

from collections import OrderedDict

from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from registry.registry import *


@RUNNERS.register_module(name="Runner")
class Runner:

    def __init__(self, 
                 model, 
                 train_dataloader, 
                 data_preprocessor, 
                 optim_wrapper,
                 custom_hooks, 
                 train_cfg):
        
        print("runer 666")


        self._hooks = []
        self.register_hooks(custom_hooks)
        

        model.setdefault('data_preprocessor', data_preprocessor)
        self.model = MODELS.build(model)


        self._train_loop = train_cfg
        self._train_dataloader = train_dataloader
        self._train_loop['dataloader_cfg'] = self._train_dataloader
        self._train_loop = self.build_train_loop(self._train_loop)

        self.optim_wrapper = optim_wrapper
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)



        # self.param_schedulers = param_scheduler


    def build_optim_wrapper(self, optim_wrapper: dict):

        optim_wrapper_cfg = copy.deepcopy(optim_wrapper)

        optim_wrapper_constructor = OPTIM_WRAPPERS.build(
            dict(type='DefaultOptimWrapperConstructor',
                optim_wrapper_cfg=optim_wrapper_cfg,))

        return optim_wrapper_constructor(self.model)
    
    # def _build_param_scheduler(self, scheduler: dict, optim_wrapper: OptimWrapper):
    #     schedulers = scheduler
    #     param_schedulers = []
    #     for scheduler in schedulers:
    #         _scheduler = copy.deepcopy(scheduler)
    #         default_end = self.max_iters
    #         _scheduler.setdefault('end', default_end)
    #         param_schedulers.append(PARAM_SCHEDULERS.build(_scheduler,
    #                 default_args=dict(
    #                 optimizer=optim_wrapper,
    #                 epoch_length=len(self.train_dataloader))))
    #     return param_schedulers
    # def build_param_scheduler(self, scheduler: dict):
    #     param_schedulers = self._build_param_scheduler(scheduler, self.optim_wrapper)
    #     return param_schedulers


    @property
    def train_loop(self):
        return self._train_loop

    def train(self) -> nn.Module:

        self.call_hook('before_run')

        model = self.train_loop.run()

        self.call_hook('after_run')

        return model
    

    def build_train_loop(self, cfg: dict):

        print(cfg)

        cfg['runner'] = self

        loop = LOOPS.build(cfg)

        return loop  

    def call_hook(self, fn_name: str, **kwargs) -> None:

        for hook in self._hooks:
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None

    def register_hook(self, hook: dict) -> None:

        print(hook)
        hook_obj = HOOKS.build(hook)
        print(hook_obj)
        self._hooks.insert(0, hook_obj)

    def register_custom_hooks(self, hook: dict) -> None:

        self.register_hook(hook)

    def register_hooks(self, custom_hooks: dict = None) -> None:
            self.register_custom_hooks(custom_hooks)






