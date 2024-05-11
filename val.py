import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs_attention as archs
# import archs
from dataset import Dataset
from metrics import iou_score, dice_coef
from utils import AverageMeter
"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='BCEDiceLoss_attention_UNet++',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    val_img_ids = img_ids[:30]
    model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    model.eval()

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        # A.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=3,#config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for image, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = image.cuda()# 已经被除了255
            target = target.cuda()# 已经被除了255 大小为[1, 1, 256, 256]
            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            dice = dice_coef(output, target)
            avg_meter.update(dice, input.size(0))

            # output = torch.sigmoid(output).cpu().numpy()

            # for i in range(len(output)):
            #     for c in range(config['num_classes']):
            #         cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
            #                     (output[i, c] * 255).astype('uint8'))

            print('dice_coef: %.4f' % avg_meter.avg)

            plot_examples(input, target, model, config['deep_supervision'], num_examples=3)
    
    torch.cuda.empty_cache()

def plot_examples(datax, datay, model, deep_supervision, num_examples=3):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18, 4*num_examples))

    for row_num in range(num_examples):
        if deep_supervision != True:
            image_arr = model(datax[row_num:row_num+1]).squeeze(0).detach().cpu().numpy()
        else:
            image_arr = model(datax[row_num:row_num + 1])[-1].squeeze(0).detach().cpu().numpy()

        ax[row_num][0].imshow(np.transpose(datax[row_num].cpu().numpy(), (1,2,0)))
        ax[row_num][0].set_title("Orignal Image")

        ax[row_num][1].imshow(np.squeeze((image_arr > 0.1)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")

        ax[row_num][2].imshow(np.transpose(datay[row_num].cpu().numpy(), (1,2,0)))
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    main()
