import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
# from pointnet2_ops import pointnet2_utils

import open3d as o3d
from pytorch3d.structures import Meshes
from mvtn.mv_run import transToMVImage
from models import models_mae
import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.misc as misc
import math
import sys
from torch.utils.tensorboard import SummaryWriter
from engine_pretrain import train_one_epoch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import datetime
import util.lr_sched as lr_sched


train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def transToMesh(points):
        batch_size = points.shape[0]
        meshs = []
        for i in range(batch_size):
            pt = points[i].cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pt)
            pcd.estimate_normals()

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1)
            vert = torch.tensor(mesh.vertices,dtype=torch.float32)
            face = torch.tensor(mesh.triangles,dtype=torch.float32)
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            # vert = torch.tensor(mesh[0].vertices,dtype=torch.float32)
            # face = torch.tensor(mesh[0].triangles,dtype=torch.float32)
            meshs.append(Meshes(
                verts=[vert],
                faces=[face] 
            ))
        return meshs

def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device('cuda')
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None



    model = models_mae.__dict__['mae_vit_base_patch16_dec512d8b'](norm_pix_loss=False)
    model.to(device)
    model_without_ddp = model
   
    print("Model = %s" % str(model_without_ddp))
    eff_batch_size = config.total_bs * 1 * misc.get_world_size()
    args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % 1)
    print("effective batch size: %d" % eff_batch_size)

   
    start_epoch = 0
    
    lr = args.blr * eff_batch_size / 256
 

    param_groups = optim_factory.add_weight_decay(model_without_ddp, 0.05)
    optimizer = torch.optim.AdamW(param_groups,lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # optimizer, scheduler = builder.build_opti_sche(base_model, config)

    
    # if args.resume:
    #     builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    # base_model.zero_grad()
    unloader = transforms.ToPILImage()
    log_writer = SummaryWriter(log_dir='./log/pretrain')
    accum_iter = 1
    args.epochs = config.max_epoch
    print(f"Start training for {config.max_epoch} epochs")
    start_time = time.time()

    for epoch in range(start_epoch, config.max_epoch + 1):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)

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
        for idx, (taxonomy_ids, model_ids, data) in enumerate(metric_logger.log_every(train_dataloader, 100, header)): #shapenet pretrain
        # for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader): #shapenet 
        # for idx, (taxonomy_ids, model_ids, data, path) in enumerate(train_dataloader): #modelnet
            # parta = path[0].split('/')[-2]
            # file_name = path[0].split('/')[-1]
            # parts = file_name.split('_')
            # partb = parts[-1].split('.')[0]
            

            # # if os.path.exists(src):
            # #     continue

            # points = data[0].cuda()
            # unloader = transforms.ToPILImage()    
            # # mesh = transToMesh(points)
            # points = points[:,:,:3]
            # images = transToMVImage(points=points,meshes=None)
            # src = 'modelnet40_normal_resampled/'+parta+'/'+parta+'_'+partb+'.jpg'
            # print(f"处理第 {idx + 1}/{len(train_dataloader)} 项",path[0])
            # image = images[0][0].cpu().clone()
            # image = image.squeeze(0)
            # image = unloader(image)
            # image.save(src)
            # continue





            #shapenet
            # continue
            # if idx % accum_iter == 0:
            #     lr_sched.adjust_learning_rate(optimizer, idx / len(train_dataloader) + epoch, args)
            # name = model_ids[0].split('.')[0]
            # npoints = config.dataset.train.others.npoints
            # dataset_name = config.dataset.train._base_.NAME
            # if dataset_name == 'ShapeNet':
            #     points = data.cuda()
            #     # image = data.cuda()
            # elif dataset_name == 'ModelNet':
            #     points = data[0].cuda()
            #     points = misc.fps(points, npoints)   
            # else:
            #     raise NotImplementedError(f'Train phase do not support {dataset_name}')


            # assert points.size(1) == npoints
            # points = train_transforms(points)
           
            
            
            # # mesh = transToMesh(points)
            # images = transToMVImage(points=points,meshes=None)
            # src = 'img/'+name+'.jpg'
            # print(f"处理第 {idx + 1}/{len(train_dataloader)} 项",src)
            # if os.path.exists(src):
            #     continue
            # image = images[0][0].cpu().clone()
            # image = image.squeeze(0)
            # image = unloader(image)
            # image.save(src)
            # # break
            
            # # gt = image[0][0].cpu().clone()
            # # gt = gt.squeeze(0)
            # # gt = unloader(gt)
            # # gt.save('image_gt.jpg')
            # continue


            image = data
            image = image.to(device, non_blocking=True)


            # continue
            with torch.cuda.amp.autocast():
                
                loss, pred, _ = model(image, mask_ratio=0.75)
            
                
            if(idx%200==0):
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
                        update_grad=(idx + 1) % accum_iter == 0)
            if (idx + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            
            metric_logger.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (idx + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((idx / len(train_dataloader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
            # forward
        
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        

        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == config.max_epoch or epoch == 1):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass