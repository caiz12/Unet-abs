""" Module for running training"""

from pkg_resources import resource_filename

import copy
import time
from collections import defaultdict


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

from unetabs import absdataset
from unetabs import spectraunet

# Making device global
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_standard(flux_file, lbls_file, outfile, testing=False, try_parallel=True):

    # Load up data
    simple_train = absdataset.AbsDataset(flux_file, lbls_file, mode='train', testing=testing)
    simple_valid = absdataset.AbsDataset(flux_file, lbls_file, mode='validate', testing=testing)

    dataloaders = {}
    dataloaders['train'] = DataLoader(simple_train, batch_size=4, shuffle=True)
    dataloaders['valid'] = DataLoader(simple_valid, batch_size=4, shuffle=True)  # , num_workers=4)

    # Setup model
    model = spectraunet.SpectraUNet()
    if device.type == 'cuda':
        if (torch.cuda.device_count() > 1) & try_parallel:
            model = nn.DataParallel(model)
    # Finish setup
    model.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

    model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=10)

    # Save!
    torch.save(model.state_dict(), outfile)


def calc_loss(pred, target, metrics, bce_weight=1.0):
    # import pdb; pdb.set_trace()
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # pred = F.sigmoid(pred)
    # dice = dice_loss(pred, target)

    loss = bce * bce_weight  # + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                # import pdb; pdb.set_trace()
                inputs = sample_batched['image']
                labels = sample_batched['labels']
                inputs = inputs.to(device)
                labels = labels.to(device)  # , torch.int64)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
                if (i_batch % 50) == 0:
                    print(i_batch)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'valid' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Command line execution
if __name__ == '__main__':

    flg=0
    flg += 2**0

    # Load data
    if flg & (2**0):
        flux_file = resource_filename('unetabs', 'data/training/simple_flux.npy')
        lbls_file = resource_filename('unetabs', 'data/training/simple_lbls.npy')
        outfile = resource_filename('unetabs', 'data/models/simple_model.pt')
        #
        train_standard(flux_file, lbls_file, outfile, testing=True)
