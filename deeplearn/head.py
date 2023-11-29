import sys


from registry.registry import *

import torch
import torch.nn as nn



@MODELS.register_module(name="LinearClsHead")
class LinearClsHead(nn.Module):
    
    def __init__(self, loss: dict, num_classes: int, in_channels: int):
        
        super(LinearClsHead, self).__init__()
        print("head")
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

        self.loss_module = MODELS.build(loss)
        print("loss")

    def pre_logits(self, feats: torch.Tensor) -> torch.Tensor:

        return feats[-1]

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """The forward process."""

        pre_logits = self.pre_logits(feats)
        print(pre_logits.size())

        # The final classification head.
        cls_score = self.fc(feats)

        print("head :", cls_score.size())


        return cls_score
    
    def loss(self, feats, data_samples, **kwargs) -> dict:


        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)

        return losses

    def _get_loss(self, cls_score: torch.Tensor, data_samples, **kwargs):

        # if 'gt_score' in data_samples[0]:
        #     target = torch.stack([i.gt_score for i in data_samples])
        # else:
        #     target = torch.cat([i.gt_label for i in data_samples])

        print(cls_score.size())

        # compute loss
        losses = dict()
        loss = self.loss_module(cls_score, data_samples, **kwargs)
        losses['loss'] = loss

        # compute accuracy
        # if self.cal_acc:
        #     acc = Accuracy.calculate(cls_score, target, topk=self.topk)
        #     losses.update({f'accuracy_top-{k}': a for k, a in zip(self.topk, acc)})

        return losses




@MODELS.register_module(name="ImageClassifier")
class ImageClassifier(BaseModel):

    def __init__(self, 
                 backbone: dict,
                 neck: dict,
                 head: dict = None,
                 data_preprocessor: dict = None,
                 init_cfg: dict = None):

        data_preprocessor = MODELS.build(data_preprocessor)

        super(ImageClassifier, self).__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)


        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
            self.with_neck = True
        else:
            self.with_neck = False
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, inputs: torch.Tensor, data_samples, mode: str = 'tensor'):

        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs, stage='neck'):

        x = self.backbone(inputs)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)

        if stage == 'neck':
            return x

        return self.head.pre_logits(x)

    def loss(self, inputs: torch.Tensor, data_samples) -> dict:

        print(inputs.size())
        feats = self.extract_feat(inputs)
        
        print(feats.size())
        return self.head.loss(feats, data_samples)

    def predict(self, inputs: torch.Tensor, data_samples, **kwargs):

        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples, **kwargs)