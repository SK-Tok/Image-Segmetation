#############################################
###### This script is made by SK-Tok ########
#############################################

# add the module path
import sys

# import torch module
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optims as optim
import torchvision.transforms as tf

# import python module
import numpy as np
import argparse
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import transform as tf
from unet import UNet
from crossentropy2d import CrossEntropy2d
from FocalLoss2d import FocalLoss2d
from poly_lr_scheduler import Poly_lr_scheduler
from seg_dataset import NPSegDataset, ImageSegDataset

# Define Argument
args = {
    'exp_name':'writing experiment name',
    'data':'writing dataset directory',
    'result':'writing output directory',
    'batch_size':'writing batch size',
    'epochs':'writing epoch number',
    'lr':'writing learning rate',
    'momentum':'writing momentum',
    'wd':'writing weight decay',
    'num_classes':'writing class number',
    'band_num':'writing band number'
}

print(args)
device = torch.device('cpu')

# Set Network
net = UNet(num_channels=3, num_classes=2)

print('select networks!!')
net.to(device)

# Main
def main(args):
    # Set parameter
    patch_size = 224
    device = torch.device('cpu')
    
    # Make Dataset
    trans = tf.Compose([
        tf.NPSegRandomCrop(patch_size),
        tf.NPSegFlip(),
        tf.NPRandomRotate()
    ])
    
    data_dir = Path(args['data'])
    train_dir = data_dir.joinpath('train')
    val_dir = data_dir.joinpath('val')
    
    # Loading Dataset
    train_dataset = NPSegDataset(train_dir.joinpath('img'), train_dir.joinpath('label'), transform=trans)
    val_dataset = NPSegDataset(val_dir.joinpath('img'), val_dir.joinpath('label'), transform=tf.NPSegRandomCrop(patch_size))
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=10)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=10)
    loaders = {'train':train_loader, 'val':val_loader}
    dataset_sizes = {'train':len(train_dataset), 'val':len(val_dataset)}
    print('Finishing Preparing Dataset!!')
    
    # Building Model
    net = UNet(num_channels=3, num_classes=2)
    print('Select Network!!')
    net.to(device)
    
    # Set Criterion and Optimizer
    criterion = CrossEntropy2d()
    optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['wd'])
    
    # Initialized History
    loss_history = {'train':[], 'val':[]
    acc_history = {'train':[], 'val':[]}
                    
    # Train the Network
    start_time = time.time()
    for epoch in range(1, args['epochs']+1):
        lr = Poly(optimizer, args['lr'], epoch-1, max_iter=args['epochs'])
        print('*'*20)
        print('Epoch{}/{}:lr={}'.format(epoch, args['epochs'],lr))
        print('*'*20)
                    
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
                    
            runnning_loss = 0.0
            running_corrects = 0.0
            
            for i, data in enumerate(loaders[phase]):
                    inputs, labels = data
                    inputs = inputs.float() / 255
                    labels = labels.long()
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    if phase == 'train':
                        outputs = net(inputs)
                    else:
                        with torch.no_grad():
                            outputs = net(inputs)
                    
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs.data, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    runnning_loss += loss.to('cpu')
                    corrects = torch.sum(preds == labels.data).to('cpu').item()
                    running_corrects += corrects / (patch_size ** 2) / args['batch_size']
                    
            epoch_loss = runnning_loss / (i+1)
            epoch_acc = ruuning_corrects / dataset_sizes[phase]
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)
            
            print('{}Loss:{:.4f}Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = net.state_dect()
    
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}h {:.0f}m'.format(elapsed_time // 3600, (elapsed_time % 3600) / 60))
    print('Best val Acc:{:.4f}'.format(best_acc))
    
    net.load_state_dict(best_mode)
    return net, loss_history, acc_history
                    
if __name__ == '__main__':
    Path(args['result']).mkdir(exist_ok = True)
    model_weight, loss_history, acc_history = main(args)
    torch.save(model_weight.state_dict(), str(Path(args['result']).joinpath('Weight.pth')))
    plt.title('Model Accuracy')
    plt.plot(np.arange(args['epochs']), acc_history['train'], label = 'train acc')
    plt.plot(np.arange(args['epochs']), acc_history['val'], label = 'val acc')
    plt.legend(loc = 'upper left')
    plt.savefig(str(Path(args['result']).joinpath('acc.png')))
    plt.title('Model Loss')
    plt.plot(np.arange(args['epochs']), acc_history['train'], label = 'train loss')
    plt.plot(np.arange(args['epochs']), acc_history['val'], label = 'val loss')
    plt.legend(loc = 'upper left')
    plt.savefig(str(Path(args['result']).joinpath('loss.png'))) 
