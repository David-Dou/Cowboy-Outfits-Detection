import os
import fire
import time
import utils
import torch
import numpy as np
from tqdm import tqdm  # plot progress bar
from config import Config
from data import CowboyDataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


import torchvision
from models import FasterRCNN_Pretained
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

opt = Config()
vis = utils.Visualizer()


def train(**kwargs):
    opt.parse(kwargs)

    # create dataloader
    # train_dataset = CowboyDataset(opt.data_dir, train=True, test=False)
    train_dataset = CowboyDataset(opt.data_dir, transforms=CowboyDataset.get_train_transform(),
                                  train=True, test=False)

    # collate_fn is used (together with) for stacking samples as a batch
    def collate_fn(batch):
        return tuple(zip(*batch))  # load every data batch as tuple

    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        collate_fn=collate_fn
    )
    # choose device
    device_selector = utils.device_selection()
    device = device_selector.device
    # create model
    model = FasterRCNN_Pretained.model
    model.to(device)
    model.train()
    # create optimizer
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # train
    total_train_loss = []
    start_time = time.time()
    for epoch in range(opt.num_epochs):
        num_iter = 1
        train_loss = []
        pgbar = tqdm(train_loader, desc='Train:')
        for img_ids, images, targets in pgbar:  # img_ids, images, targets of a batch of images
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            loss_dict = model(images, targets)  # faster_rcnn from torchvision uses list parameters
            # print(loss_dict)
            losses = sum([loss for loss in loss_dict.values()])  # loss of one image?
            losses_value = losses.item()  # .item() to get 1*1 tensor's value
            train_loss.append(losses_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            num_iter += 1
            pgbar.set_description(desc=f'Epoch {epoch+1}, Batch {num_iter}, Loss={losses_value}')

            if (epoch + 1) % opt.model_save_freq == 0:
                model_name = time.strftime('checkpoints/fasterrcnn_resnet50_fpn_pretrained_%m%d_%H_%M_%S.pth')
                torch.save(model.state_dict(), model_name)

        epoch_train_loss = np.mean(np.array(train_loss))
        total_train_loss.append(epoch_train_loss)
        end_time = time.time()

        print(f'Epoch Completed: {epoch + 1}/{opt.num_epochs}, Time: {end_time - start_time}, '
              f'Train Loss: {epoch_train_loss}')

    vis.plot_with_pyplot(list(range(1, opt.num_epochs + 1)), total_train_loss,
                         xlabel="epoch", ylabel="train_loss",
                         xlim=[1, opt.num_epochs],
                         legend=["train_loss"],
                         title="train_loss = " + str(total_train_loss[-1]),
                         yscale="linear")


def test(**kwargs):
    opt.parse(kwargs)

    # create dataloader
    test_dataset = CowboyDataset(data_dir=opt.data_dir, transforms=CowboyDataset.get_test_transform(),
                                 train=False, test=True)

    def collate_fn(batch):
        return tuple(zip(*batch))
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=False,  # ?
        collate_fn=collate_fn
    )
    # choose device
    # load model, move this part to models/ ?
    device_selector = utils.device_selection()
    device = device_selector.device
    # create model and load checkpoints
    # create a Faster R-CNN model without pre-training
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                 pretrained_backbone=False)  # torchvision.models.detection contains mobilenet
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_class = len(test_dataset.categories) + 1
    # replace the pre-trained model's head with a new one, use FastRCNNPredictor here
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=num_class)
    # load the trained weights
    model.load_state_dict(state_dict=torch.load(opt.load_model_path))
    # move model to the right device
    model.to(device)
    model.eval()
    # predict
    score_threshold = 0.5
    img_outputs = []

    pgbar = tqdm(test_dataloader, desc='Test')
    for (img_ids, images) in pgbar:
        images = [image.to(device) for image in images]
        outputs = model(images)

        for img_id, output in zip(img_ids, outputs):
            bboxes = output['boxes'].data.cpu().numpy()
            labels = output['labels'].data.cpu().numpy()
            scores = output['scores'].data.cpu().numpy()

            # filter bboxes with low confidence scores
            mask = scores >= score_threshold
            bboxes = bboxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            bboxes, labels, scores = bboxes.tolist(), labels.tolist(), scores.tolist()
            img_outputs.append((img_id, bboxes, labels, scores))  # annotation information of one image

            # img_name = test_dataset.id2name[img_id]
            # img = plt.imread(os.path.join('data/cowboyoutfits/images', img_name))
            # fig = plt.imshow(img)
            # label_names = [test_dataset.categories[test_dataset.label2cat[label]] for label in labels]
            # vis.show_bboxes(fig.axes, bboxes, label_names)
            # plt.show()

    model_name = opt.load_model_path.split('.')[0].split('/')[1]
    file_name = f'{model_name}@{int(score_threshold*100)}.json'
    test_dataset._results2json(img_outputs, os.path.join('results', file_name))

    # with torch.no_grad():  # sentences under this statement won't be used in backpropogation
    #     valid_loss = []
    #
    #     for (img_ids, images, targets) in

if __name__ == '__main__':
    fire.Fire()