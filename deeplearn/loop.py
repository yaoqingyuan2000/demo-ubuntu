

from registry.registry import *


def collate(batch):

    inputs = []

    data_samples = []

    labels = []

    for item in batch:

        inputs.append(item['inputs'])
        data_samples.append (item['data_samples'])
        labels.append(item['labels'])

    return inputs, labels, data_samples


def build_dataloader(cfg: dict) -> DataLoader:

    dataloader_cfg = copy.deepcopy(cfg)

    # build dataset
    dataset_cfg = dataloader_cfg.pop('dataset')

    print(dataset_cfg)

    dataset = DATASETS.build(dataset_cfg)


    batch_size = dataloader_cfg.get('batch_size')
    num_workers = dataloader_cfg.get('num_workers'),

    # build sampler
    # sampler_cfg = dataloader_cfg.pop('sampler')
    # sampler = DATA_SAMPLERS.build()

    # worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
    # worker_init_fn_type = worker_init_fn_cfg.pop('type')
    # worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
   
    # collate_fn_cfg = dataloader_cfg.pop('collate_fn', dict(type='pseudo_collate'))
    # collate_fn_type = collate_fn_cfg.pop('type')
    # collate_fn = FUNCTIONS.get(collate_fn_type)
    
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    return data_loader




@LOOPS.register_module(name="Loop")
class Loop():

    def __init__(self, 
                 runner, 
                 dataloader_cfg: dict, 
                 max_epochs: int, 
                 val_begin: int = 1, 
                 val_interval: int = 1,) -> None:

        self.dataloader = build_dataloader(dataloader_cfg)
        self.runner =  runner

        self._max_epochs = int(max_epochs)

        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval

        self.stop_training = False


    def run(self) -> torch.nn.Module:

        # self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:

            self.run_epoch()

            # self._decide_current_val_interval()
            # if (self.runner.val_loop is not None
            #         and self._epoch >= self.val_begin
            #         and self._epoch % self.val_interval == 0):
            #     self.runner.val_loop.run()

        # self.runner.call_hook('after_train')

        return self.runner.model

    def run_epoch(self) -> None:

        # self.runner.call_hook('before_train_epoch')

        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):

            data = dict(inputs=data_batch[0],labels=data_batch[1])

            self.run_iter(idx, data)

        # self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: dict) -> None:

        # self.runner.call_hook('before_train_iter', batch_idx=idx, data_batch=data_batch)

        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(data_batch, optim_wrapper=self.runner.optim_wrapper)

        print("loss:", outputs)

        # self.runner.call_hook('after_train_iter', batch_idx=idx, data_batch=data_batch, outputs=outputs)
        
        self._iter += 1