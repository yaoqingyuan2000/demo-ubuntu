

from registry.registry import *




@OPTIM_WRAPPERS.register_module(name="OptimWrapper")
class OptimWrapper():

    def __init__(self, optimizer: Optimizer, accumulative_counts: int = 1):
        
        self._accumulative_counts = accumulative_counts

        self.optimizer = optimizer

        self._inner_count = 0

        self._max_counts = -1

        self._remainder_counts = -1

    def should_update(self) -> bool:

        return (self._inner_count % self._accumulative_counts == 0
                or self._inner_count == self._max_counts)

    def update_params(self, 
                      loss: 
                      torch.Tensor, 
                      step_kwargs: dict = None,
                      zero_kwargs: dict = None) -> None:
 
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        loss.backward(**kwargs)
        self._inner_count += 1

    def zero_grad(self, **kwargs) -> None:
        self.optimizer.zero_grad(**kwargs)

    def step(self, **kwargs) -> None:

        self.optimizer.step(**kwargs)

    def initialize_count_status(self, model: nn.Module, init_counts: int, max_counts: int) -> None:

        self._inner_count = init_counts
        self._max_counts = max_counts

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:

        print("        loss:", loss)

        if self._accumulative_counts == 1:
            # update parameters without gradient accumulation. The gradient
            # should not be rescaled and `loss_factor=1`.
            loss_factor = 1
        elif self._max_counts == -1:
            loss_factor = self._accumulative_counts
        else:

            if self._inner_count < self._max_counts - self._remainder_counts:
                loss_factor = self._accumulative_counts
            else:
                loss_factor = self._remainder_counts
            assert loss_factor > 0, (
                'loss_factor should be larger than zero! This error could '
                'happened when initialize_iter_status called with an '
                'error `init_counts` or `max_counts`')

        loss = loss / loss_factor
        return loss

    @property
    def inner_count(self):
        """Get the number of updating parameters of optimizer wrapper."""
        return self._inner_count

    def __repr__(self):
        wrapper_info = (f'Type: {type(self).__name__}\n'
                        f'_accumulative_counts: {self._accumulative_counts}\n'
                        'optimizer: \n')
        optimizer_str = repr(self.optimizer) + '\n'
        return wrapper_info + optimizer_str
    

@OPTIM_WRAPPERS.register_module(name="DefaultOptimWrapperConstructor")
class DefaultOptimWrapperConstructor():

    def __init__(self, optim_wrapper_cfg: dict,):

        self.optim_wrapper_cfg = optim_wrapper_cfg.copy()
        self.optimizer_cfg = self.optim_wrapper_cfg.pop('optimizer')

    def __call__(self, model: nn.Module) -> OptimWrapper:
        
        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        
        optimizer_cfg = self.optimizer_cfg.copy()

        optimizer_cfg['params'] = model.parameters()

        optimizer = OPTIMIZERS.build(optimizer_cfg)

        optim_wrapper_cfg['optimizer'] = optimizer

        optim_wrapper = OPTIM_WRAPPERS.build(optim_wrapper_cfg)

        return optim_wrapper

