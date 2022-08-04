import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms


class BBoxUtility(object):
    def __init__(self):
        pass

    #-------------------------------------#
    #   调整先验框并将xywh 转换为 x1y1x2y2
    #-------------------------------------#
    def decode_boxes(self, pred_box, anchors, variances = [0.1, 0.2]):
        """
        pred_box: 预测值 [18525, 4]
        anchors:  先验框 [18525, 4]
        """
        #---------------------------------------------------------#
        #   anchors[:, :2] 先验框中心       预测框坐标 = 先验框坐标 + 预测值 * variances * 先验框宽高
        #   anchors[:, 2:] 先验框宽高       预测框宽高 = 先验框宽高 * e^预测值
        #   对先验框的中心和宽高进行调整，获得预测框
        #---------------------------------------------------------#
        boxes = torch.cat((anchors[:, :2] + pred_box[:, :2] * variances[0] * anchors[:, 2:],
                           anchors[:, 2:] * torch.exp(pred_box[:, 2:] * variances[1])), 1)

        #---------------------------------------------------------#
        #   获得左上角和右下角
        #---------------------------------------------------------#
        boxes[:, :2] -= boxes[:, 2:] / 2    # xy - 1/2wh = x1y1
        boxes[:, 2:] += boxes[:, :2]        # x1y1 + wh  = x2y2
        return boxes

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        use_batch = True
        if box_a.dim() == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]

        n = box_a.size(0)
        A = box_a.size(1)
        B = box_b.size(1)

        max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2), box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
        min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2), box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, :, 0] * inter[:, :, :, 1]

        area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter

        out = inter / area_a if iscrowd else inter / union
        return out if use_batch else out.squeeze(0)

    def fast_non_max_suppression(self, box_thre, class_thre, mask_thre, nms_iou=0.5, top_k=200, max_detections=100):
        """快速非极大值抑制

        Args:
            box_thre (tensor):   保存的框   [43, 4]     注意下面的43都是num_of_kept_boxes
            class_thre (tensor): 保存的类别 [43, 80]
            mask_thre (tensor):  保存的分割 [43, 32]
            nms_iou (float, optional): 非极大值抑制阈值,越小越严格. Defaults to 0.5.
            top_k (int, optional): 对每一个种类单独进行排序的最终保留值. Defaults to 200.
            max_detections (int, optional): 最大检测数量. Defaults to 100.

        Returns:
            
        """
        #---------------------------------------------------------#
        #   先进行tranpose，方便后面的处理
        #   [43, 80] -> [80, 43]
        #---------------------------------------------------------#
        class_thre      = class_thre.transpose(1, 0).contiguous()
        #---------------------------------------------------------#
        #   每一行坐标为该种类所有的框的得分，
        #   对每一个种类单独进行排序
        #   class_thre/idx: [80, 43]
        #---------------------------------------------------------#
        class_thre, idx = class_thre.sort(1, descending=True)

        #---------------------------------------------------------#
        #   保留前top_k个值
        #---------------------------------------------------------#
        idx             = idx[:, :top_k].contiguous()   # [80, 43]
        class_thre      = class_thre[:, :top_k]         # [80, 43]

        # 80, 43
        num_classes, num_dets = idx.size()
        #---------------------------------------------------------#
        #   将num_classes作为第一维度，对每一个类进行非极大抑制
        #   [43, 4]  -> [80*43, 4]  -> [80, 43, 4]
        #   [43, 32] -> [80*43, 32] -> [80, 43, 32]
        #
        #       box_thre: [43, 4]
        #       idx.view(-1): [80*43]
        #       box_thre[idx.view(-1), :]  不理解这样取数据不会报错吗
        #---------------------------------------------------------#
        box_thre    = box_thre[idx.view(-1), :].view(num_classes, num_dets, 4)
        mask_thre   = mask_thre[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou         = self.jaccard(box_thre, box_thre)
        #---------------------------------------------------------#
        #   [80, 43, 43]
        #   取矩阵的上三角部分
        #---------------------------------------------------------#
        iou.triu_(diagonal=1)
        iou_max, _  = iou.max(dim=1)

        #---------------------------------------------------------#
        #   获取和高得分重合程度比较低的预测结果
        #---------------------------------------------------------#
        keep        = (iou_max <= nms_iou)
        class_ids   = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)    # [80, 43]

        box_nms     = box_thre[keep]        # [80, 43, 4]  -> [837, 4]
        class_nms   = class_thre[keep]      # [80, 43]     -> [837]
        class_ids   = class_ids[keep]       # [80, 43]     -> [837]
        mask_nms    = mask_thre[keep]       # [80, 43, 32] -> [837, 32]

        #  [837]
        _, idx      = class_nms.sort(0, descending=True)
        idx         = idx[:max_detections]  # [837] -> [100]
        box_nms     = box_nms[idx]          # [837, 4] -> [100, 4]
        class_nms   = class_nms[idx]        # [837] -> [100]
        class_ids   = class_ids[idx]        # [837] -> [100]
        mask_nms    = mask_nms[idx]         # [837, 32] -> [100, 32]
        return box_nms, class_nms, class_ids, mask_nms

    def traditional_non_max_suppression(self, box_thre, class_thre, mask_thre, pred_class_max, nms_iou, max_detections):

        num_classes     = class_thre.size()[1]
        pred_class_arg  = torch.argmax(class_thre, dim = -1)

        box_nms, class_nms, class_ids, mask_nms = [], [], [], []
        for c in range(num_classes):
            #--------------------------------#
            #   取出属于该类的所有框的置信度
            #   判断是否大于门限
            #--------------------------------#
            c_confs_m = pred_class_arg == c
            if len(c_confs_m) > 0:
                #-----------------------------------------#
                #   取出得分高于confidence的框
                #-----------------------------------------#
                boxes_to_process = box_thre[c_confs_m]
                confs_to_process = pred_class_max[c_confs_m]
                masks_to_process = mask_thre[c_confs_m]
                #-----------------------------------------#
                #   进行iou的非极大抑制
                #-----------------------------------------#
                idx         = nms(boxes_to_process, confs_to_process, nms_iou)
                #-----------------------------------------#
                #   取出在非极大抑制中效果较好的内容
                #-----------------------------------------#
                good_boxes  = boxes_to_process[idx]
                confs       = confs_to_process[idx]
                labels      = c * torch.ones((len(idx))).long()
                good_masks  = masks_to_process[idx]
                box_nms.append(good_boxes)
                class_nms.append(confs)
                class_ids.append(labels)
                mask_nms.append(good_masks)
        box_nms, class_nms, class_ids, mask_nms = torch.cat(box_nms, dim = 0), torch.cat(class_nms, dim = 0), \
                                                  torch.cat(class_ids, dim = 0), torch.cat(mask_nms, dim = 0)

        idx = torch.argsort(class_nms, 0, descending=True)[:max_detections]
        box_nms, class_nms, class_ids, mask_nms = box_nms[idx], class_nms[idx], class_ids[idx], mask_nms[idx]
        return box_nms, class_nms, class_ids, mask_nms

    def yolact_correct_boxes(self, boxes, image_shape):
        """将预测框还原到原图尺寸

        Args:
            boxes (tensor):      保存下来的预测框 [num_of_kept_boxes, 4]
            image_shape (array): 原图尺寸,如 [1330, 1330]

        Returns:
            boxes: 扩展到原图尺寸的预测框 [num_of_kept_boxes, 4]
        """
        # [1330, 1330]
        image_size          = np.array(image_shape)[::-1]
        image_size          = torch.tensor([*image_size]).type(boxes.dtype).cuda() if boxes.is_cuda else torch.tensor([*image_size]).type(boxes.dtype)

        # [1330, 1330, 1330, 1330]
        scales              = torch.cat([image_size, image_size], dim=-1)
        # 扩展到原图尺寸
        boxes               = boxes * scales
        boxes[:, [0, 1]]    = torch.min(boxes[:, [0, 1]], boxes[:, [2, 3]])
        boxes[:, [2, 3]]    = torch.max(boxes[:, [0, 1]], boxes[:, [2, 3]])
        boxes[:, [0, 1]]    = torch.max(boxes[:, [0, 1]], torch.zeros_like(boxes[:, [0, 1]]))
        boxes[:, [2, 3]]    = torch.min(boxes[:, [2, 3]], torch.unsqueeze(image_size, 0).expand([boxes.size()[0], 2]))
        return boxes

    def crop(self, masks, boxes):
        """通过boxes剪裁masks

        Args:
            masks (tensor): 还原到原图尺寸的mask [1330, 1330, num_of_kept_boxes]
            boxes (tensor): 还原到原图尺寸的框   [num_of_kept_boxes, 4]

        Returns:
            masks: 剪裁后的masks
        """
        h, w, n     = masks.size()  # 1330, 1330, 11
        # x1 x2 y1 y2: [11]
        x1, x2      = boxes[:, 0], boxes[:, 2]
        y1, y2      = boxes[:, 1], boxes[:, 3]

        # rows cols: [1330, 1330, 11]
        rows        = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
        cols        = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

        #--------------------------#
        #   限制mask的左右上下范围
        #   四个值全是True或者False,形状都是[1330, 1330, 11]
        #--------------------------#
        masks_left  = rows >= x1.view(1, 1, -1) # x1.view(1, 1, -1)作用 [11] -> [1, 1, 11]
        masks_right = rows < x2.view(1, 1, -1)
        masks_up    = cols >= y1.view(1, 1, -1)
        masks_down  = cols < y2.view(1, 1, -1)
        #--------------------------#
        #   True/False相乘,结果还是True/False
        #   形状是[1330, 1330, 11]
        #--------------------------#
        crop_mask   = masks_left * masks_right * masks_up * masks_down
        #--------------------------#
        #   每个mask的像素都乘以True/False
        #   [1330, 1330, 11] * [1330, 1330, 11] = [1330, 1330, 11]
        #--------------------------#
        return masks * crop_mask.float()

    def decode_nms(self, outputs, anchors, confidence, nms_iou, image_shape, traditional_nms=False, max_detections=100):
        #---------------------------------------------------------#
        #   pred_box:   [b, 18525, 4]       对应每个先验框的调整情况
        #   pred_class: [b, 18525, 81]      对应每个先验框的种类
        #   pred_mask:  [b, 18525, 32]      对应每个先验框的语义分割情况
        #   pred_proto: [b, 136, 136, 32]   对P3进行上采样,调整通道为32,需要和结合pred_mask使用
        #   去除batch=1
        #---------------------------------------------------------#
        pred_box    = outputs[0].squeeze()
        pred_class  = outputs[1].squeeze()
        pred_masks  = outputs[2].squeeze()
        pred_proto  = outputs[3].squeeze()

        #---------------------------------------------------------#
        #   调整先验框并将xywh 转换为 x1y1x2y2
        #   [18525, 4] -> [18525, 4]
        #   boxes是左上角、右下角的形式。
        #---------------------------------------------------------#
        boxes       = self.decode_boxes(pred_box, anchors)

        #---------------------------------------------------------#
        #   pred_class的最后维度再模型中经过了softmax处理
        #   除去背景的部分，并获得最大的得分
        #   [18525, 80]
        #   [18525]
        #---------------------------------------------------------#
        pred_class          = pred_class[:, 1:]             # [18525, 81] -> [18525, 80]
        pred_class_max, _   = torch.max(pred_class, dim=1)  # 返回值和下标    [18525]
        keep        = (pred_class_max > confidence)         # True/False     [18525]

        #---------------------------------------------------------#
        #   保留满足得分的框，如果没有框保留，则返回None
        #---------------------------------------------------------#
        box_thre    = boxes[keep, :]        # [num_of_kept_boxes, 4]
        class_thre  = pred_class[keep, :]   # [num_of_kept_boxes, 80]
        mask_thre   = pred_masks[keep, :]   # [num_of_kept_boxes, 32]
        if class_thre.size()[0] == 0:
            return None, None, None, None, None

        if not traditional_nms:
            #-------------------------------#
            #   box_thre:   [100, 4]
            #   class_thre: [100]
            #   class_ids:  [100]
            #   mask_thre:  [100, 32]
            #-------------------------------#
            box_thre, class_thre, class_ids, mask_thre = self.fast_non_max_suppression(box_thre, class_thre, mask_thre, nms_iou)
            keep        = class_thre > confidence   # [100] True / False
            box_thre    = box_thre[keep]            # [100, 4]  -> [11, 4]
            class_thre  = class_thre[keep]          # [100]     -> [11]
            class_ids   = class_ids[keep]           # [100]     -> [11]
            mask_thre   = mask_thre[keep]           # [100, 32] -> [11, 32]
        else:
            box_thre, class_thre, class_ids, mask_thre = self.traditional_non_max_suppression(box_thre, class_thre, mask_thre, pred_class_max[keep], nms_iou, max_detections)

        #---------------------------------------------------------#
        #   将预测框还原到原图尺寸
        #   box_thre: [11, 4]
        #---------------------------------------------------------#
        box_thre    = self.yolact_correct_boxes(box_thre, image_shape)

        #---------------------------------------------------------#
        #   P3上采样的分割 * 预测的分割
        #   pred_proto      [136, 136, 32]  P3进行上采样并调整维度
        #   mask_thre       [num_of_kept_boxes, 32]
        #   [136, 136, 32]@[32, num_of_kept_boxes] = [136, 136, num_of_kept_boxes]
        #   masks_sigmoid   [136, 136, num_of_kept_boxes]
        #---------------------------------------------------------#
        masks_sigmoid   = torch.sigmoid(torch.matmul(pred_proto, mask_thre.t()))
        #----------------------------------------------------------------------#
        #   对mask处理并防止超出预测框边界
        #   [136, 136, num_of_kept_boxes] -> [num_of_kept_boxes, 136, 136] ->
        #   [num_of_kept_boxes, 1330, 1330] -> [1330, 1330, num_of_kept_boxes]
        #----------------------------------------------------------------------#
        masks_sigmoid   = masks_sigmoid.permute(2, 0, 1).contiguous()
        masks_sigmoid   = F.interpolate(masks_sigmoid.unsqueeze(0), (image_shape[0], image_shape[1]), mode='bilinear', align_corners=False).squeeze(0)
        masks_sigmoid   = masks_sigmoid.permute(1, 2, 0).contiguous()
        #------------------------------------------#
        #   防止mask超出预测框边界
        #   [1330, 1330, num_of_kept_boxes]
        #------------------------------------------#
        masks_sigmoid   = self.crop(masks_sigmoid, box_thre)

        #----------------------------------------------------------------------#
        #   masks_arg: [1330, 1330, num_of_kept_boxes] -> [1330, 1330]
        #   获得每个像素点所属的实例(最大值)
        #----------------------------------------------------------------------#
        masks_arg       = torch.argmax(masks_sigmoid, dim=-1)
        #----------------------------------------------------------------------#
        #   masks_arg   [1330, 1330, num_of_kept_boxes]
        #   判断每个像素点是否满足门限需求
        #----------------------------------------------------------------------#
        masks_sigmoid   = masks_sigmoid > 0.5

        return box_thre, class_thre, class_ids, masks_arg, masks_sigmoid

