
from registry.registry import *

@METRICS.register_module()
class Accuracy():

    def __init__(self,) -> None:
        super().__init__()
        self.topk = (1, ), 
        self.thrs = (0.5, )

    def process(self, data_batch: dict):
        pass


    def compute_metrics(self, results):

        metrics = {}

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])
        acc = self.calculate(pred, target, self.topk, self.thrs)

        multi_thrs = len(self.thrs) > 1
        for i, k in enumerate(self.topk):
            for j, thr in enumerate(self.thrs):
                name = f'top{k}'
                if multi_thrs:
                    name += '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                metrics[name] = acc[i][j].item()


        return metrics

    @staticmethod
    def calculate(pred, target, topk, thrs):
  
        # For pred score, calculate on all topk and thresholds.
        pred = pred.float()
        target = target.to(torch.int64)

        num = pred.size(0)

        maxk = max(topk)

        if maxk > pred.size(1):
            raise ValueError(
                f'Top-{maxk} accuracy is unavailable since the number of '
                f'categories is {pred.size(1)}.')

        pred_score, pred_label = pred.topk(maxk, dim=1)
        pred_label = pred_label.t()
        correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
        results = []
        for k in topk:
            results.append([])
            for thr in thrs:

                correct = correct & (pred_score.t() > thr)
                
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)

                acc = correct_k.mul_(100. / num)
                results[-1].append(acc)

            return results