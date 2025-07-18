import torch
from unet import UNet
from torch.utils.data import DataLoader
from dataset import ISBI2012Dataset
from metric import StreamSegMetrics
import argparse
import u_transform as ut


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='isbi2012',
                        choices=['voc', 'cityscapes','isbi2012'], help='Name of dataset')
    parser.add_argument("--model", type=str, default='unet',
                         help='model name')
    parser.add_argument("--crop_size", type=int, default=512)
    return parser

def validate(opts, model, loader, device, metrics):
    model.eval()
    metrics.reset()
        
    #验证开始
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
    return score

if __name__ == '__main__':
    opts = get_argparser().parse_args()
    img_save_dir = 'outputs'
    weight_path = 'checkpoints/best_unet_isbi2012_os16.pth'
    # params
    batch_size = 1
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #metric
    metrics = StreamSegMetrics(2)
    # dataset and dataload
    val_transform = ut.ExtCompose([
            ut.ExtResize(opts.crop_size),
            ut.ExtToTensor(),
        ])
    val_dst = ISBI2012Dataset(root=opts.data_root,image_set='test',transform=val_transform)
    val_loader = DataLoader(val_dst, batch_size=batch_size, shuffle=False)
    # model
    model = UNet(1, 1,bilinear=True,decoderFp=opts.val_fp,encoderFp=opts.val_fp).to(device)  
    model.load_state_dict(torch.load(weight_path)['model_state'])
    #validate
    val_score = validate(opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
    #print result
    print(metrics.to_str(val_score))

