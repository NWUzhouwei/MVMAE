# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
from mvtn.mv_run import transToMVImage
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
        # i, (targets, meshes, points)
    # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (samples, mesh, points) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images = transToMVImage(points=points,mesh=mesh,batch_size=points.shape[0])
        images[:, 1:8] = torch.rot90(images[:, 1:8], k=2, dims=[-2,-1])
        x_list = torch.split(images, 1, dim=1)
        x_list = [torch.squeeze(t, dim=1) for t in x_list]
        row_list = [torch.cat(x_list[i:i+4], dim=3) for i in range(0, 16, 4)]
        image = torch.cat(row_list, dim=2)

        
        unloader = transforms.ToPILImage()
        gt = image[0][0].cpu().clone()
        gt = gt.squeeze(0)
        gt = unloader(gt)
        gt.save('image_gt.jpg')
        image = image.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, _ = model(image, mask_ratio=args.mask_ratio)

        if(data_iter_step%100==0):
            print(pred)
            gt = image[0][0].cpu().clone()
            gt = gt.squeeze(0)
            gt = unloader(gt)
            gt.save('image_gt.jpg')
            pred = pred[0][0].cpu().clone()
            pred = pred.squeeze(0)
            pred = unloader(pred)
            pred.save('image_pred.jpg')

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}