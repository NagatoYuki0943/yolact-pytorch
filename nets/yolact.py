import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import ResNet


#------------------------------------------#
#   单方向FPN,和YoloV3很像,只有上采样过程
#------------------------------------------#
class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # [512, 1024, 2048]
        self.in_channels = in_channels

        #------------------------------------------#
        #   C3、C4、C5通道数均调整成256
        #   1x1Conv
        #   C5: [b, 2048, 17, 17] -> [b, 256, 17, 17]
        #   C4: [b, 1024, 34, 34] -> [b, 256, 34, 34]
        #   C3: [b, 512 , 68, 68] -> [b, 256, 68, 68]
        #------------------------------------------#
        self.lat_layers     = nn.ModuleList(
            [
                nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels
            ]
        )

        #------------------------------------------#
        #   特征融合后用于进行特征整合
        #   3x3Conv + ReLU
        #   C5: [b, 256, 17, 17] -> [b, 256, 17, 17]
        #   C4: [b, 256, 34, 34] -> [b, 256, 34, 34]
        #   C3: [b, 256, 68, 68] -> [b, 256, 68, 68]
        #------------------------------------------#
        self.pred_layers    = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)) for _ in self.in_channels
            ]
        )

        #------------------------------------------#
        #   对P5进行下采样获得P6和P7
        #   3x3Conv s=2 + ReLU
        #   P5toP6: [b, 256, 17, 17] -> [b, 256, 9, 9]
        #   P6toP7: [b, 256, 9, 9] -> [b, 256, 5, 5]
        #------------------------------------------#
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True)
                )
            ]
        )

    def forward(self, backbone_features):
        #------------------------------------------#
        #   C3、C4、C5通道数均调整成256
        #   1x1Conv
        #   C5: [b, 2048, 17, 17] -> [b, 256, 17, 17]
        #   C4: [b, 1024, 34, 34] -> [b, 256, 34, 34]
        #   C3: [b, 512 , 68, 68] -> [b, 256, 68, 68]
        #------------------------------------------#
        P5          = self.lat_layers[2](backbone_features[2])
        P4          = self.lat_layers[1](backbone_features[1])
        P3          = self.lat_layers[0](backbone_features[0])

        #------------------------------------------#
        #   [b, 256, 17, 17] -> [b, 256, 34, 34] + [b, 256, 34, 34] = [b, 256, 34, 34]
        #------------------------------------------#
        P5_upsample = F.interpolate(P5, size=(backbone_features[1].size()[2], backbone_features[1].size()[3]), mode='nearest')
        P4          = P4 + P5_upsample

        #------------------------------------------#
        #   [b, 256, 34, 34] -> [b, 256, 68, 68] + [b, 256, 68, 68] = [b, 256, 68, 68]
        #------------------------------------------#
        P4_upsample = F.interpolate(P4, size=(backbone_features[0].size()[2], backbone_features[0].size()[3]), mode='nearest')
        P3          = P3 + P4_upsample

        #------------------------------------------#
        #   特征融合后用于进行特征整合
        #   3x3Conv + ReLU
        #   P5: [b, 256, 17, 17] -> [b, 256, 17, 17]
        #   P4: [b, 256, 34, 34] -> [b, 256, 34, 34]
        #   P3: [b, 256, 68, 68] -> [b, 256, 68, 68]
        #------------------------------------------#
        P5 = self.pred_layers[2](P5)
        P4 = self.pred_layers[1](P4)
        P3 = self.pred_layers[0](P3)

        #------------------------------------------#
        #   3x3Conv s=2 + ReLU
        #   对P5进行下采样获得P6和P7
        #   P5toP6: [b, 256, 17, 17] -> [b, 256, 9, 9]
        #   P6toP7: [b, 256, 9, 9] -> [b, 256, 5, 5]
        #------------------------------------------#
        P6 = self.downsample_layers[0](P5)
        P7 = self.downsample_layers[1](P6)

        return P3, P4, P5, P6, P7


#------------------------------------------#
#   原型 prototype mask
#   对P3进行上采样,调整通道为32,需要和结合pred_mask使用
#   [b, 256, 68, 68] -> [b, 32, 136, 136]
#------------------------------------------#
class ProtoNet(nn.Module):
    def __init__(self, coef_dim):   # coef_dim=32
        super().__init__()
        # [b, 256, 68, 68] -> [b, 256, 68, 68]
        self.proto1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # [b, 256, 68, 68] -> [b, 256, 136, 136]
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # [b, 256, 136, 136] -> [b, 32, 136, 136]
        self.proto2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, coef_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # [b, 256, 68, 68] -> [b, 32, 136, 136]
        x = self.proto1(x)
        x = self.upsample(x)
        x = self.proto2(x)
        return x


#--------------------------------#
#   P3,P4,P5,P6,P7使用同一个头
#   用于获取每一个有效特征层
#   获得对应的预测结果
#--------------------------------#
class PredictionModule(nn.Module):
    def __init__(self, num_classes, coef_dim=32, aspect_ratios = [1, 1 / 2, 2]):
        super().__init__()
        self.num_classes    = num_classes   # 81
        self.coef_dim       = coef_dim

        # [b, 256, h, w] -> [b, 256, h, w]
        self.upfeature = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        #-------------------------------------#
        #   框框xywh
        #   [b, 256, h, w] -> [b, 3*4, h, w]
        #-------------------------------------#
        self.bbox_layer = nn.Conv2d(256, len(aspect_ratios) * 4, kernel_size=3, padding=1)
        #-------------------------------------#
        #   classes
        #   [b, 256, h, w] -> [b, 3*81, h, w]
        #-------------------------------------#
        self.conf_layer = nn.Conv2d(256, len(aspect_ratios) * self.num_classes, kernel_size=3, padding=1)
        #-------------------------------------#
        #   mask置信度
        #   [b, 256, h, w] -> [b, 3*32, h, w]
        #-------------------------------------#
        self.coef_layer = nn.Sequential(
            nn.Conv2d(256, len(aspect_ratios) * self.coef_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        bs      = x.size(0)
        # [b, 256, h, w] -> [b, 256, h, w]
        x       = self.upfeature(x)
        #--------------------------------------------------------------------------#
        #   框框xywh
        #   [b, 256, h, w] -> [b, 3*4, h, w] -> [b, h, w, 3*4] -> [b, h*w*3, 4]
        #--------------------------------------------------------------------------#
        box     = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(bs, -1, 4)
        #--------------------------------------------------------------------------#
        #   classes
        #   [b, 256, h, w] -> [b, 3*81, h, w] -> [b, h, w, 3*81] -> [b, h*w*3, 81]
        #--------------------------------------------------------------------------#
        conf    = self.conf_layer(x).permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes)
        #--------------------------------------------------------------------------#
        #   mask
        #   [b, 256, h, w] -> [b, 3*32, h, w] -> [b, h, w, 3*32] -> [b, h*w*3, 32]
        #--------------------------------------------------------------------------#
        coef    = self.coef_layer(x).permute(0, 2, 3, 1).reshape(bs, -1, self.coef_dim)
        #---------------------------#
        #   box:  [b, h*w*3, 4]   box
        #   conf: [b, h*w*3, 81]  种类置信度
        #   coef: [b, h*w*3, 32]  mask置信度
        #---------------------------#
        return box, conf, coef


class Yolact(nn.Module):
    def __init__(self, num_classes, coef_dim=32, pretrained=False, train_mode=True):    # num_classes = 80 + 1
        super().__init__()
        #------------------------------------------#
        #   主干特征提取网络获得三个初步特征
        #   C3: [b, 512 , 68, 68]
        #   C4: [b, 1024, 34, 34]
        #   C5: [b, 2048, 17, 17]
        #------------------------------------------#
        self.backbone               = ResNet(layers=[3, 4, 6, 3])
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/resnet50_backbone_weights.pth"))

        #------------------------------------------#
        #   构建特征金字塔，获得五个有效特征层
        #   P3: [b, 256, 68, 68]
        #   P4: [b, 256, 34, 34]
        #   P5: [b, 256, 17, 17]
        #   P6: [b, 256,  9,  9]
        #   P7: [b, 256,  5,  5]
        #------------------------------------------#
        self.fpn                    = FPN([512, 1024, 2048])

        #------------------------------------------#
        #   对P3进行上采样,调整通道为32,需要和结合pred_mask使用
        #   [b, 256, 68, 68] -> [b, 32, 136, 136]
        #------------------------------------------#
        self.proto_net              = ProtoNet(coef_dim=coef_dim)
        #------------------------------------------#
        #   用于获取每一个有效特征层(P3,P4,P5,P6,P7)
        #   获得对应的预测结果
        #   box:  [b, h*w*3, 4]
        #   conf: [b, h*w*3, 81]
        #   coef: [b, h*w*3, 32]
        #------------------------------------------#
        self.prediction_layers      = PredictionModule(num_classes, coef_dim=coef_dim)

        #------------------------------------------#
        #   P3分割mask
        #   [b, 256, 68, 68] -> [b, 81-1, 68, 68]
        #------------------------------------------#
        self.semantic_seg_conv      = nn.Conv2d(256, num_classes - 1, kernel_size=1)

        self.train_mode             = train_mode

    def forward(self, x):
        #------------------------------------------#
        #   主干特征提取网络获得三个初步特征
        #   C3: [b, 512 , 68, 68]
        #   C4: [b, 1024, 34, 34]
        #   C5: [b, 2048, 17, 17]
        #------------------------------------------#
        features = self.backbone(x)
        #------------------------------------------#
        #   构建特征金字塔，获得五个有效特征层
        #   P3: [b, 256, 68, 68]
        #   P4: [b, 256, 34, 34]
        #   P5: [b, 256, 17, 17]
        #   P6: [b, 256,  9,  9]
        #   P7: [b, 256,  5,  5]
        #------------------------------------------#
        features = self.fpn.forward(features)
        #---------------------------------------------------#
        #   原型mask
        #   对P3进行上采样,调整通道为32,需要和结合pred_mask使用
        #   [b, 256, 68, 68] -> [b, 32, 136, 136] -> [b, 136, 136, 32]
        #---------------------------------------------------#
        pred_proto = self.proto_net(features[0])
        pred_proto = pred_proto.permute(0, 2, 3, 1).contiguous()

        #--------------------------------------------#
        #   将5个特征层利用同一个head的预测结果堆叠
        #   box:  [b, h*w*3, 4]
        #   conf: [b, h*w*3, 81]
        #   coef: [b, h*w*3, 32]
        #   (68^2 + 34^2 + 17^2 + 9^2 + 5^2) * 3 = 18525
        #   pred_boxes:     [b, 18525,  4]      对应每个先验框的调整情况
        #   pred_classes:   [b, 18525, 81]      对应每个先验框的种类
        #   pred_masks:     [b, 18525, 32]      对应每个先验框的语义分割情况置信度
        #--------------------------------------------#
        pred_boxes, pred_classes, pred_masks = [], [], []
        for f_map in features:
            box_p, class_p, mask_p = self.prediction_layers(f_map)
            pred_boxes.append(box_p)
            pred_classes.append(class_p)
            pred_masks.append(mask_p)
        pred_boxes      = torch.cat(pred_boxes,   dim=1)
        pred_classes    = torch.cat(pred_classes, dim=1)
        pred_masks      = torch.cat(pred_masks,   dim=1)

        if self.train_mode:
            #----------------------------------------------#
            #   P3分割mask
            #   P3: [b, 256, 68, 68] -> [b, 81-1, 68, 68]
            #----------------------------------------------#
            pred_segs   = self.semantic_seg_conv(features[0])
            return pred_boxes, pred_classes, pred_masks, pred_proto, pred_segs
        else:
            # softmax
            pred_classes = F.softmax(pred_classes, -1)
            return pred_boxes, pred_classes, pred_masks, pred_proto
