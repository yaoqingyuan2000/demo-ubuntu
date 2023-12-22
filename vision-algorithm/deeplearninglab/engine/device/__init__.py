# Copyright (c) OpenMMLab. All rights reserved.
from .utils import (get_device, get_max_cuda_memory, is_cuda_available,
                    is_dipu_available, is_mlu_available, is_mps_available,
                    is_npu_available, is_npu_support_full_precision)

from .type import (IS_MLU_AVAILABLE, IS_MPS_AVAILABLE, 
                    IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE)

__all__ = [
    'get_max_cuda_memory', 'get_device', 'is_cuda_available',
    'is_mlu_available', 'is_mps_available', 'is_npu_available',
    'is_dipu_available', 'is_npu_support_full_precision',
    'IS_CUDA_AVAILABLE', 'IS_MLU_AVAILABLE', 'IS_MPS_AVAILABLE', 
    'IS_NPU_AVAILABLE'
]
