# import network
import os
import random
import argparse
import numpy as np
from torchvision.transforms import transforms
from torch.utils import data
from dataset import ISBI2012Dataset
import u_transform as ut
from metric import StreamSegMetrics
from unet import UNet
import torch
import torch.nn as nn
# from visualizer import Visualizer
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import time


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='isbi2012',
                        choices=['voc', 'cityscapes','isbi2012'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='unet',
                         help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_epoch", type=int, default=1200,
                        help="epoch number (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='CosineAnnealingLR', choices=['poly', 'step','ReduceLROnPlateau','CosineAnnealingLR'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=True,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 8)')
    parser.add_argument("--val_batch_size", type=int, default=8,
                        help='batch size for validation (default: 8)')
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='binary_cross_entropy',
                        choices=['cross_entropy', 'focal_loss','dice_loss','binary_cross_entropy'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=2e-5,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    
    train_transform = ut.ExtCompose([
        ut.ExtResize(size=opts.crop_size),
        ut.ExtRandomScale((0.5, 2.0)),
        ut.ExtRandomCrop(size=(opts.crop_size-256, opts.crop_size-256), pad_if_needed=True),
        ut.ExtRandomAffine(),
        ut.ExtRandomHorizontalFlip(),
        ut.ExtRandomVerticalFlip(),
        ut.ExtRandomAdjustSharpness(),#useful
        ut.ExtGaussianBlur(3,(0.1,1.0),0.3),
        ut.ExtToTensor(),
    ])
    if opts.crop_val:
        val_transform = ut.ExtCompose([
            ut.ExtResize(size=opts.crop_size),
            ut.ExtCenterCrop(opts.crop_size),
            ut.ExtToTensor()
        ])
    else:
        val_transform = ut.ExtCompose([
            ut.ExtResize(size=opts.crop_size),
            ut.ExtToTensor(),
        ])
    if opts.dataset == 'isbi2012':
        train_dst = ISBI2012Dataset(root=opts.data_root,image_set='train',transform=train_transform,augment=True)
        val_dst = ISBI2012Dataset(root=opts.data_root,image_set='test',transform=val_transform)
    
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics,epoch, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists('results/%d'%epoch):
            os.mkdir('results/%d'%epoch)
        # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
        #                            std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if opts.val_fp==16:

                images = images.to(device, dtype=torch.float16)
            elif opts.val_fp == 32:
                images = images.to(device,dtype=torch.float32)
            
            labels = labels.to(device, dtype=torch.long)
            if opts.val_fp==16:
                with torch.autocast(device_type='cuda',dtype=torch.float16):
                    outputs = model(images)
            elif opts.val_fp == 32:
                outputs = model(images)
            outputs = torch.sigmoid(outputs).float()
            outputs[outputs>=0.5] = 1
            outputs[outputs<0.5] = 0
            preds = outputs.detach().cpu().numpy()
            
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
        torch.cuda.empty_cache()
        score = metrics.get_results()

    return score, ret_samples

def training(opts,model,train_loader,device,criterion,optimizer,scheduler,epoch,epochs,lossArr):
    model.train()
    losses = []

    for i,(images, labels) in enumerate(train_loader):
        
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(images)
        outputs = outputs.squeeze()
        loss = criterion(outputs ,labels.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        if i % 50 == 0 and i > 0:
            print(f'epoch:{epoch}/{epochs}, iter:{i}th, train loss:{loss.item()}')
        torch.cuda.empty_cache()
    mean_loss = sum(losses) / len(losses)
    lossArr.append(mean_loss)
    print(f"loss at epoch {epoch} is {mean_loss}")
    scheduler.step(mean_loss)
def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower()=='isbi2012':
        opts.num_classes = 2

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1
    elif opts.dataset == 'isbi2012':
        opts.val_batch_size=1


    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    if opts.model=='unet':
        model = UNet(n_channels=1,n_classes=1,bilinear=True)
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr,momentum=0.99,weight_decay=  1e-4)# 正则化系数λ
    if opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True  # 5个epoch检查一次loss，触发后打印出lr
    )
    elif opts.lr_policy == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=opts.total_epoch,
            eta_min=1e-8
        )

    if opts.train_fp == 16:
        scaler = torch.cuda.amp.GradScaler(enabled=(opts.train_fp == 16))
    # Set up criterion
    weight = torch.tensor([10]).to(device=device)
    if opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'binary_cross_entropy':
        criterion = nn.BCEWithLogitsLoss(weight=weight)
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        # model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, epoch=-1, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    model.train()
    lossArr = []
    avg_train_time = 0.0
    avg_val_time = 0.0
    val_time = 0
    weight_path = 'checkpoints/latest_unet_isbi2012_os16.pth'
    for epoch in range(opts.total_epoch):
        # =====  Train  =====
        train_start = time.time()
        training(opts,model,train_loader,device,criterion,optimizer,scheduler,epoch,opts.total_epoch,lossArr=lossArr)
        
        one_epoch_time = time.time()-train_start
        avg_train_time+=one_epoch_time
        save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
        if epoch % 1 == 0 and epoch > 0:
                print("validation...")
                val_start = time.time()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    epoch=epoch,ret_samples_ids=vis_sample_id)
                one_val_time = time.time()-val_start
                avg_val_time+=one_val_time
                val_time+=1
                print(metrics.to_str(val_score))
                
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))
    avg_train_time/=opts.total_epoch
    avg_val_time/=val_time
    print(f"train time per epoch:{avg_train_time*1000:.2f}毫秒")
    print(f"val time per epoch:{avg_val_time*1000:.2f}毫秒")
    plt.figure(figsize=(10, 6))
    plt.plot(lossArr, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('results/loss.png' , bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()