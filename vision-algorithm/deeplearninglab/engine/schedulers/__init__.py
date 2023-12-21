# Copyright (c) OpenMMLab. All rights reserved.
from .optimizer import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIMIZERS,
                        AmpOptimWrapper, ApexOptimWrapper, BaseOptimWrapper,
                        DefaultOptimWrapperConstructor, OptimWrapper,
                        OptimWrapperDict, ZeroRedundancyOptimizer,
                        build_optim_wrapper)
# yapf: disable
from .scheduler import (ConstantLR, ConstantMomentum, ConstantParamScheduler,
                        CosineAnnealingLR, CosineAnnealingMomentum,
                        CosineAnnealingParamScheduler, ExponentialLR,
                        ExponentialMomentum, ExponentialParamScheduler,
                        LinearLR, LinearMomentum, LinearParamScheduler,
                        MultiStepLR, MultiStepMomentum,
                        MultiStepParamScheduler, OneCycleLR,
                        OneCycleParamScheduler, PolyLR, PolyMomentum,
                        PolyParamScheduler, ReduceOnPlateauLR,
                        ReduceOnPlateauMomentum, ReduceOnPlateauParamScheduler,
                        StepLR, StepMomentum, StepParamScheduler,
                        _ParamScheduler)

# yapf: enable
__all__ = [
    'OPTIM_WRAPPER_CONSTRUCTORS', 'OPTIMIZERS', 'build_optim_wrapper',
    'DefaultOptimWrapperConstructor', 'ConstantLR', 'CosineAnnealingLR',
    'ExponentialLR', 'LinearLR', 'MultiStepLR', 'StepLR', 'ConstantMomentum',
    'CosineAnnealingMomentum', 'ExponentialMomentum', 'LinearMomentum',
    'MultiStepMomentum', 'StepMomentum', 'ConstantParamScheduler',
    'CosineAnnealingParamScheduler', 'ExponentialParamScheduler',
    'LinearParamScheduler', 'MultiStepParamScheduler', 'StepParamScheduler',
    '_ParamScheduler', 'OptimWrapper', 'AmpOptimWrapper', 'ApexOptimWrapper',
    'OptimWrapperDict', 'OneCycleParamScheduler', 'OneCycleLR', 'PolyLR',
    'PolyMomentum', 'PolyParamScheduler', 'ReduceOnPlateauLR',
    'ReduceOnPlateauMomentum', 'ReduceOnPlateauParamScheduler',
    'ZeroRedundancyOptimizer', 'BaseOptimWrapper'
]

# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
from .lr_scheduler import (ConstantLR, CosineAnnealingLR, CosineRestartLR,
                           ExponentialLR, LinearLR, MultiStepLR, OneCycleLR,
                           PolyLR, ReduceOnPlateauLR, StepLR)
from .momentum_scheduler import (ConstantMomentum, CosineAnnealingMomentum,
                                 CosineRestartMomentum, ExponentialMomentum,
                                 LinearMomentum, MultiStepMomentum,
                                 PolyMomentum, ReduceOnPlateauMomentum,
                                 StepMomentum)
from .param_scheduler import (ConstantParamScheduler,
                              CosineAnnealingParamScheduler,
                              CosineRestartParamScheduler,
                              ExponentialParamScheduler, LinearParamScheduler,
                              MultiStepParamScheduler, OneCycleParamScheduler,
                              PolyParamScheduler,
                              ReduceOnPlateauParamScheduler,
                              StepParamScheduler, _ParamScheduler)

# yapf: enable
__all__ = [
    'ConstantLR', 'CosineAnnealingLR', 'ExponentialLR', 'LinearLR',
    'MultiStepLR', 'StepLR', 'ConstantMomentum', 'CosineAnnealingMomentum',
    'ExponentialMomentum', 'LinearMomentum', 'MultiStepMomentum',
    'StepMomentum', 'ConstantParamScheduler', 'CosineAnnealingParamScheduler',
    'ExponentialParamScheduler', 'LinearParamScheduler',
    'MultiStepParamScheduler', 'StepParamScheduler', '_ParamScheduler',
    'PolyParamScheduler', 'PolyLR', 'PolyMomentum', 'OneCycleParamScheduler',
    'OneCycleLR', 'CosineRestartParamScheduler', 'CosineRestartLR',
    'CosineRestartMomentum', 'ReduceOnPlateauParamScheduler',
    'ReduceOnPlateauLR', 'ReduceOnPlateauMomentum'
]