import torch
from torch import nn
import torch.functional as F    
import math
import cv2 as cv
import numpy as np

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                            (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU 
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU: 
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area 
    return iou  # IoU

class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)




class YOLOV8DetectionLoss:

    def __init__(self):  # model must be de-paralleled
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        self.stride = 32

        self.reg_max = 16

        self.device = "cpu"

        self.topk=10 
        self.alpha=0.5
        self.beta=6.0

        self.num_classes = 8

        self.eps = 1e-9

        self.bbox_loss = BboxLoss(self.reg_max - 1, True).to(self.device)

    def __call__(self, preds, batch):

        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        

        # dtype = pred_scores.dtype
        # batch_size = pred_scores.shape[0]

        # imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        
        # anchor_points, stride_tensor = self.make_anchors(feats, self.stride, 0.5)

        # # targets
        # targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        # targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        
        # gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # gt_mask = gt_bboxes.sum(2, keepdim=True).gt_(0)


        # # pboxes
        # pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        self.stride = preds["strides"]

        feats = preds["feats"]
        

        anchor_points, stride_tensor = self.make_anchors(feats, self.stride, 0.5)
        gt_labels = batch["gt_labels"]
        gt_bboxes = batch["gt_bboxes"]
        pred_scores = preds["pd_scores"]
        pred_bboxes = preds["pd_bboxes"]

        gt_mask = batch["gt_mask"]

        print("anchor_points", anchor_points.size())


        self.image = batch["image"]

        self.max_num_obj = gt_bboxes.size(1)
        self.bs = 2
        self.num_anchors = 8400

        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(pred_scores,
                                                                    pred_bboxes,
                                                                    anchor_points * stride_tensor,
                                                                    gt_labels,
                                                                    gt_bboxes,
                                                                    gt_mask)
        
        target_scores_sum = max(target_scores.sum(), 1)

        dtype = pred_scores.dtype

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        print("cls loss:", loss[1])

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain


    def assigner(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, gt_mask):

        
        self.n_max_boxes = gt_bboxes.size(1)
        self.num_max_boxes = self.n_max_boxes
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))


        an_mask = self.get_anchors_mask(anchor_points, gt_bboxes)

        print("anchor_mask:", an_mask.size())

        # temp_mask = an_mask * gt_mask
        # for point in anchor_points * temp_mask[0, 1].view(-1, 1):
        #     cv.circle(self.image, (int(point[0]), int(point[1])), 3, (255, 0, 0), 3)
        # cv.namedWindow('Image')
        # cv.imshow('Image', self.image) 
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        overlaps, align_metric = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, an_mask * gt_mask)

        print("align_metric:", align_metric.size())
        print("overlaps:", overlaps.size())


        # for box in gt_bboxes[0]:
        #     cv.rectangle(self.image, (int(box[0].item()), int(box[1].item())), 
        #                              (int(box[2].item()), int(box[3].item())), (0, 255, 0), 2, 8)

        # temp_mask = an_mask * gt_mask

        # for i, metric in enumerate(align_metric[0]):
        #     for j, score in enumerate(metric):
        #         if score.item() > 0.:
        #             print(score.item())
        #             cv.circle(self.image, (int(anchor_points[j, 0]), int(anchor_points[j, 1])), 3, (255, 0, 0), 1)
        # cv.namedWindow('Image1')
        # cv.imshow('Image1', self.image) 

        # for i, iou in enumerate(overlaps[0]):
        #     for j, score in enumerate(iou):
        #         if score.item() > 0.:
        #             print(score.item())
        #             cv.circle(self.image, (int(anchor_points[j, 0]), int(anchor_points[j, 1])), 3, (255, 0, 0), 1)
        # cv.namedWindow('Image2')
        # cv.imshow('Image2', self.image)

        # cv.waitKey(0)
        # cv.destroyAllWindows()

        topk_mask = gt_mask.expand(-1, -1, self.topk).bool()
        topk_mask = self.select_topk_candidates(align_metric, topk_mask)
        print("topk_mask:", topk_mask.size())

        mask = gt_mask * an_mask * topk_mask

        print("mask:", mask.size())

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(overlaps, mask, self.max_num_obj)

        print("fg_mask:", fg_mask.size())

        target_labels, target_bboxes, target_scores = self.get_target(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        print("target_labels:", target_labels.size())
        print("target_bboxes:", target_bboxes.size())


        # Normalize
        align_metric *= mask_pos
        # print(align_metric)
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        # print((align_metric * pos_overlaps) / (pos_align_metrics + self.eps))
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        print("target_scores:", target_scores.size())

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


    def get_target(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]

        target_gt_idx = target_gt_idx + batch_ind * self.num_max_boxes  # (b, h*w)

        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)

        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores


    def select_highest_overlaps(self, overlaps, mask_pos, num_max_boxes):

        fg_mask = mask_pos.sum(-2)

        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes

            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)  # (b, n_max_boxes, h*w)
            
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        return target_gt_idx, fg_mask, mask_pos


    def select_topk_candidates(self, align_metric, topk_mask):
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(align_metric, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(align_metric.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        # filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)
        count_tensor.to(align_metric.dtype)
        return count_tensor

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask):

        num_anchors = pd_scores.shape[-2]
        mask = mask.bool()
        ind = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.num_max_boxes)  # b, max_num_obj   
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj

        # 预测分数
        bbox_scores = torch.zeros([self.bs, self.num_max_boxes, num_anchors], dtype=pd_scores.dtype, device=pd_scores.device)
        bbox_scores[mask] = pd_scores[ind[0], :, ind[1]][mask]  # b, max_num_obj, h*w

        print("mask:", mask.size())
        print("bbox_scores:", bbox_scores.size())

        print("pd_bboxes", pd_bboxes.size())
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.num_max_boxes, -1, -1)[mask]

        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, num_anchors, -1)[mask]

        overlaps = torch.zeros([self.bs, self.num_max_boxes, num_anchors], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        # 
        overlaps[mask] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return overlaps, align_metric

    def get_anchors_mask(self, anchor_points, gt_bboxes):

        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((anchor_points - lt, rb - anchor_points), dim=2)\
            .view(self.bs, self.max_num_obj, self.num_anchors, -1)

        mask = bbox_deltas.amin(3).gt_(self.eps)

        return mask

    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij') 
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)





# %%

num_class = 8
batch_size = 2


pd_bboxes = torch.randint(0, 650, (batch_size, 80 * 80 + 40 * 40 + 20 * 20, 4))
pd_bboxes = pd_bboxes.float()
pd_scores = torch.rand((batch_size, 80 * 80 + 40 * 40 + 20 * 20, num_class))


gt_labels = torch.randint(0, num_class, (2, 3, 1))
gt_labels[0,2,0] = 0


gt_bboxes = torch.randint(0, 320, (batch_size, 3, 4))
gt_bboxes[0,0] = torch.tensor([280, 280, 360, 360])
gt_bboxes[0,1] = torch.tensor([10, 10, 260, 260])
gt_bboxes[0,2] = torch.tensor([10, 10, 100, 100])

gt_bboxes[1,0] = torch.tensor([230, 230, 370, 370])
gt_bboxes[1,1] = torch.tensor([17, 17, 270, 270])
gt_bboxes[1,2] = torch.tensor([307, 17, 357, 57])

gt_mask = torch.ones((2, 3, 1))
gt_mask[0,2,0] = 0


width = 640
height = 640
image = np.zeros((height, width, 3), np.uint8)
input = torch.rand((batch_size, 3, 640, 640))

print("input:", input.size())

print("gt_labels:", gt_labels.size())
# print(gt_labels)
print("gt_bboxes:", gt_bboxes.size())
# print(gt_bboxes)

print("gt_mask:", gt_mask.size())

print("pd_scores:", pd_scores.size())
print("pd_bboxes:", pd_bboxes.size())



for box in (gt_bboxes * gt_mask)[0]:
    cv.rectangle(image, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), (0, 0, 255), 2)
# cv.namedWindow('Image')
# cv.imshow('Image', image) 
# cv.waitKey(0)
# cv.destroyAllWindows()

feats = [torch.rand((batch_size, 256, 80, 80)), 
         torch.rand((batch_size, 512, 40, 40)), 
         torch.rand((batch_size, 512, 20, 20))]
s = 640
strides = torch.tensor([s / x.shape[-2] for x in feats])  # forward


preds = {"pd_scores": pd_scores, "pd_bboxes": pd_bboxes,
         "feats": feats, "strides": strides}
batch = {"gt_labels": gt_labels, "gt_bboxes": gt_bboxes, "gt_mask": gt_mask, "image": image}

loss = YOLOV8DetectionLoss()
loss(preds, batch)




