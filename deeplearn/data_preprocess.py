
import torch
import torch.nn as nn

from registry.registry import *



@MODELS.register_module(name="DataPreprocessor")
class DataPreprocessor(nn.Module):

    def __init__(self):
        super().__init__()

        self.device = 'cpu'

        print("data preprocessor")

    def cast_data(self, data: dict):


        input = torch.tensor(data['inputs']).permute(0, 3, 1, 2)
        label = torch.tensor(data['labels'])

        data['inputs'] = input.float()
        data['data_samples'] = label.long()

        data.pop('labels')

        data['inputs'].to(self.device)
        
        return data

    def forward(self, data: dict, training: bool = False):
     
        return self.cast_data(data)
    
    def to(self, *args, **kwargs) -> nn.Module:

        return super().to(*args, **kwargs)




class BaseModel(nn.Module):
    
    def __init__(self, data_preprocessor, init_cfg: dict = None):
        super().__init__()

        self.data_preprocessor = data_preprocessor

    def init_weights(self):
        pass

    def train_step(self, data: dict, optim_wrapper) -> dict:

        data = self.data_preprocessor(data, True)

        losses = self._run_forward(data, mode='loss')

        parsed_losses, log_vars = self.parse_losses(losses)

        print(parsed_losses)
        print(log_vars)

        optim_wrapper.update_params(parsed_losses)

        return log_vars

    def _run_forward(self, data: dict, mode: str) -> dict:

        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results
    
    def parse_losses(self, losses: dict):

        log_vars = []
        for loss_name, loss_value in losses.items():

            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])


        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])

        return loss, log_vars