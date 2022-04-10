import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# print(torchvision.__path__)


class FasterRCNN_Pretained:
    num_class = 6
    # get number of input features for the classifier
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # print(model.roi_heads.box_predictor)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained model's head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=num_class)
    # print(model.roi_heads.box_predictor)
