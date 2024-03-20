import pickle
import click

from rich.progress import track
from matplotlib import pyplot as plt

import torch as th
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset.coco import create_coco_dataset
from dataset.dataset import HumanFacesDataset

from utilities.utils import *

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='Enable debug mode', default=False)
@click.pass_context
def router_cmd(ctx: click.Context, debug):
    ctx.obj['debug_mode'] = debug
    invoked_subcommand = ctx.invoked_subcommand
    if invoked_subcommand is None:
        logger.info('No subcommand was specified')
    else:
        logger.info(f'Invoked subcommand: {invoked_subcommand}')
    
@router_cmd.command()
@click.option('--path2data', help='Path to source data', required=True)
def setup_coco(path2data):
    logger.info('Setting coco file up...')
    create_coco_dataset(path2data)
    
@router_cmd.command()
@click.option('--path2source_imgs', help='Path to image source', required=True)
@click.option('--path2data', help='Path to data', default='data/')
@click.option('--annotation_file', help='Path to coco file', default='dataset/coco.json')
def grabber(path2source_imgs, path2data, annotation_file):
    logger.info('Grabbing all data...')
    if not os.path.exists(path2data):
        os.makedirs(path2data)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = HumanFacesDataset(path2source_imgs, annotation_file, transforms=transform)
    with open(os.path.join(path2data, 'train_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
        
    logger.info('Train dataset saved!')
    
@router_cmd.command()
@click.option('--path2data', help='path to data', type=click.Path(True), default='data/train_dataset.pkl')
@click.option('--path2models', help='Path to models', default='models/')
@click.option('--path2metrics', help='Path to metrics', default='metrics/')
@click.option('--num_epochs', help='Number of epochs', default=10)
@click.option('--bt_size', help='Batch size', default=32)
def train(path2data, path2models, path2metrics, num_epochs, bt_size):
    logger.debug('Training...')
    if not os.path.exists(path2models):
        os.makedirs(path2models)    
    if not os.path.exists(path2metrics):
        os.makedirs(path2metrics)
    
    with open(path2data, 'rb') as f:
        dataset = pickle.load(f)
    train_loader = DataLoader(dataset, batch_size=bt_size, shuffle=True, collate_fn=collate_fn)
    nb_data = len(train_loader)
    print(nb_data)
    
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    
    pretrained_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    pretrained_model = pretrained_model.to(device)
    
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    in_features = pretrained_model.roi_heads.box_predictor.cls_score.in_features
    pretrained_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    pretrained_model = pretrained_model.to(device)

    parameters = [p for p in pretrained_model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)

    all_losses = []
    all_loss_dict = []
    for epoch in range(num_epochs):
        pretrained_model.train()
        for _, (imgs, annotations) in track(enumerate(train_loader)):
            inputs = list(img.to(device) for img in imgs)
            labels = [{k:v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict: dict = pretrained_model(inputs, labels)
            losses = sum(loss for loss in loss_dict.values())
            to_add_loss_dict = {k: v.cpu().item() for k, v in loss_dict.items()}
            loss_value = losses.cpu().item()
            all_losses.append(loss_value)
            all_loss_dict.append(to_add_loss_dict)
            
            if not th.isfinite(losses).all():
                logger.error(f'Loss is not finite. Loss: {losses}')
                continue
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
        
        print(f'Epoch: {epoch + 1}/{num_epochs}, lr: {lr_scheduler}, Loss: {loss_value}')

    th.save(pretrained_model.cpu(), os.path.join(path2models, 'best.pth'))
    logger.info('The model was saved ...!')
    
    plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(path2metrics, 'training_loss.png'))
    plt.show()
    
@router_cmd.command()
@click.option('--path2models', help='path to models', type=click.Path(True), default='models/')
@click.option('--bt_size', help='Batch size', default=64)
def inference(path2models, bt_size):
    logger.debug('Inference...')
    
if __name__ == '__main__':
    logger.info('...')
    router_cmd(obj={})