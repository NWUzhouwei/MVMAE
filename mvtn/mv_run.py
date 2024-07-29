import torch
import torch.nn as nn
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable


import scipy
import scipy.spatial


from tqdm import tqdm
import pickle as pkl


import torchvision.transforms as transforms
import torchvision

import argparse
import numpy as np
import time
import os


from mvtn.mvtn_util import *
from mvtn.ops import *

from mvtn.view_selector import MVTN
from mvtn.multi_view import *
# from mvtn.mvtn import *
# from mvtn.renderer import *
from mvtn.mvrenderer import *
import open3d as o3d



# from torch.utils.tensorboard import SummaryWriter

loader = transforms.Compose([
    transforms.ToTensor()])
unloader = transforms.ToPILImage()

PLOT_SAMPLE_NBS = [242, 7, 549, 112, 34]


parser = argparse.ArgumentParser(description='MVTN-PyTorch')
parser.add_argument('--data_dir', default='/home/liu/DataSet/ModelNet40/',  help='path to 3D dataset')
parser.add_argument('--gpu', type=int,
                    default=0, help='GPU number ')
parser.add_argument('--pc_rendering', dest='pc_rendering', default=True,
                    action='store_true', help='use point cloud renderer instead of mesh renderer')
parser.add_argument('--epochs', default=1, type=int,
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', '-b', default=3, type=int,
                    help='mini-batch size (default: 20)')
parser.add_argument('--config_file', '-cfg',  default="config.yaml", help='the conifg yaml file for more options.')
parser.add_argument('--views_config', '-s',  default="spherical", choices=["circular", "random", "learned_circular", "learned_direct", "spherical", "learned_spherical", "learned_random", "learned_transfer", "custom"],
                    help='the selection type of views ')
parser.add_argument('--nb_views', type=int,default=64,
                    help='number of views in the multi-view setup')

args = parser.parse_args()
args = vars(args)
config = read_yaml(args["config_file"]) #从config.yaml中读取
setup = {**args, **config}
initialize_setup(setup)


torch.cuda.set_device(int(setup["gpu"]))
cudnn.benchmark = True


mvtn = MVTN(nb_views=setup["nb_views"]).cuda()
mvrenderer = MVRenderer(nb_views=setup["nb_views"], return_mapping=True).cuda()


models_bag = {  "mvtn": mvtn, "mvrenderer": mvrenderer}


def transToMVImage(points,meshes):
    batch_size = points.shape[0]
    point = []
    for i in range(batch_size):
        pt = points[i].cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pt)
        centroid = np.mean(np.asarray(pcd.points), axis=0)
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) - centroid)

        # 计算点云的最大距离
        max_distance = np.max(np.linalg.norm(np.asarray(pcd.points), axis=1))

        # 将点云归一化到(-1, 1)范围
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) / max_distance)
        point.append(pcd.points)
    point = torch.tensor(point, device='cuda')










    azim, elev, dist = models_bag["mvtn"](
        point, c_batch_size=len(points))
    aaa = time.time()
    rendered_images = models_bag["mvrenderer"](
        points=point,meshes=meshes,azim=azim, elev=elev, dist=dist)
    # rendered_images, _, _, _ = models_bag["mvrenderer"](
    #     meshes, points, azim=azim, elev=elev, dist=dist)
    bbb = time.time()
    x_list = torch.split(rendered_images, 1, dim=1)
    x_list = [torch.squeeze(t, dim=1) for t in x_list]
    row_list = [torch.cat(x_list[i:i+8], dim=3) for i in range(0, 64, 8)]
    image = torch.cat(row_list, dim=2)
    return image

