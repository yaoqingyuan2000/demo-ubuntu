
class Registry:

    def __init__(self, name: str, bulid_func = None):

        self._name = name
        self._module_dict = dict()

        if bulid_func is None:
            self.bulid_func = build_from_cfg
        else:
            self.bulid_func = bulid_func

    def __len__(self):
        return len(self._module_dict)
    
    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self,module, module_name: str = None) -> None:


        if module_name is None:
            module_name = module.__name__

        self._module_dict[module_name] = module


    def register_module(self, name=None, module=None):

        if module is not None:
            self._register_module(module=module, module_name=name)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name)
            return module

        return _register
    
    def build(self, cfg: dict):

        return self.bulid_func(cfg, registry=self)
    
    def get(self, key: str):

        obj_cls = self._module_dict[key]
        
        return obj_cls

def build_from_cfg(cfg: dict, registry: Registry):

    args = cfg.copy()

    obj_type = args.pop('type')

    obj_cls = registry.get(obj_type)

    obj = obj_cls(**args) 

    return obj

def build_model_from_cfg(cfg: dict, registry: Registry):

    return build_from_cfg(cfg, registry)



def build_scheduler_from_cfg(cfg: dict, registry: Registry, default_args: dict = None):

    return build_from_cfg(cfg, registry)

RUNNERS = Registry('runner')

LOOPS = Registry('loop')

# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook')

# manage data-related modules
DATASETS = Registry('dataset')

FUNCTIONS = Registry('function')

TRANSFORMS = Registry('transform')

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', build_model_from_cfg)

# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer')
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry('optim_wrapper')

# PARAM_SCHEDULERS = Registry('parameter scheduler', build_func=build_scheduler_from_cfg)

METRICS = Registry('metric')
# manage evaluator
EVALUATOR = Registry('evaluator')

# manage visualizer
VISUALIZERS = Registry('visualizer')


def register_torch_optimizers():

    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module(module=_optim)
            torch_optimizers.append(module_name)

    return torch_optimizers

def register_torch_loss():
    torch_loss = []
    for module_name in dir(torch.nn.modules.loss):
        if module_name.startswith('__'):
            continue
        loss = getattr(torch.nn.modules.loss, module_name)
        if inspect.isclass(loss) and issubclass(loss, torch.nn.modules.Module):
            if loss.__name__ in ['CrossEntropyLoss']:
                MODELS.register_module(module=loss)
                torch_loss.append(module_name)
                
    return torch_loss

TORCH_OPTIMIZERS = register_torch_optimizers()

TORCH_LOSS = register_torch_loss()


