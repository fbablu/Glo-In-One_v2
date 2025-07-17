from model_loader import load_model_checkpoint
from util.image_pool import ImagePool
import os
from sklearn.metrics import f1_score, confusion_matrix
from engine import Engine
from math import ceil
from sklearn import metrics
import loss_functions.loss_2D as loss
from tensorboardX import SummaryWriter
import timeit
import random
from unet2D_ns_for_Glo_v2 import UNet2D as UNet2D_scale
from MOTSDataset_2D_Patch_supervise_normal_csv_Glo_v2 import MOTSValDataSet_normal as MOTSValDataSet_joint
import os.path as osp
import skimage
from matplotlib import cm
from scipy.ndimage import morphology
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import scipy.misc
import torch.optim as optim
import cv2
import pickle
import numpy as np
from torch.utils import data
import torch.nn as nn
import torch
import glob
import argparse
import os
import sys
import pandas as pd

sys.path.append("/Data/DoDNet/")


# from MOTSDataset_2D_Patch_normal import MOTSDataSet, MOTSValDataSet, my_collate
# from MOTSDataset_2D_Patch_supervise_csv import MOTSValDataSet as MOTSValDataSet_joint
# from MOTSDataset_2D_Patch_supervise_normal_csv_512 import MOTSValDataSet as MOTSValDataSet_joint


# from apex import amp
# from apex.parallel import convert_syncbn_model
# from focalloss import FocalLoss2dff

start = timeit.default_timer()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def one_hot_2D(targets, C=2):
    targets_extend = targets.clone()
    targets_extend.unsqueeze_(1)
    one_hot = torch.zeros(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3),
                          device=targets.device, dtype=torch.float32)
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


# def one_hot_2D(targets, C=2):
#     targets_extend = targets.clone()
#     targets_extend.unsqueeze_(1)  # convert to Nx1xHxW
#     one_hot = torch.cuda.FloatTensor(targets_extend.size(
#         0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
#     one_hot.scatter_(1, targets_extend, 1)
#     return one_hot


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():

    parser = argparse.ArgumentParser(description="Glo_v2")

    # parser.add_argument("--valset_dir", type=str, default='KI_data_testingset_demo/data_list.csv')
    # parser.add_argument("--valset_dir", type=str, default='./data/HC_data_patch/ddd/data_list.csv')
    # parser.add_argument("--valset_dir", type=str, default='./data/Human_Patches/test/data_list.csv')
    # parser.add_argument("--valset_dir", type=str, default='./data/Mice_Patches/test/data_list.csv')
    # parser.add_argument("--valset_dir", type=str, default='./data/Human_Patches/test/data_list.csv')
    # parser.add_argument("--valset_dir", type=str,
    #                     default='/data2/4yue/8-ADE_patch_300x300/data_list.csv')
    # parser.add_argument("--valset_dir", type=str,
    #                     default='/data2/4yue/workshop2020_patch_300x300/data_list.csv')
    parser.add_argument("--valset_dir", type=str,
                        default='Test_Patch/data_list.csv')
    # parser.add_argument("--valset_dir", type=str, default='/Data2/KI_data_validationset_patch/')
    parser.add_argument("--snapshot_dir", type=str,
                        default='snapshots_2D/fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.05_0.05_normalwhole_0907/')
    # parser.add_argument("--reload_path", type=str, default='snapshots_2D/fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.05_0.05_normalwhole_0907/MOTS_DynConv_fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.05_0.05_normalwhole_0907_e74.pth')
    # parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/fold1_with_white/')
    # parser.add_argument("--reload_path", type=str, default='snapshots_2D/final/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_final_e199.pth')
    # parser.add_argument("--reload_path", type=str, default='snapshots_2D/final/For_Hum/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_e136.pth')
    # parser.add_argument("--reload_path", type=str,
    #                     default='snapshots_2D/final/Mice_fix/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_e184.pth')
    parser.add_argument("--reload_path", type=str,
                        default='weights/Glo_v2_segmentation_model.pth')
    # parser.add_argument("--reload_path", type=str, default='snapshots_2D/final/For_Mice/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_e176.pth')

    ##################################################################################################################################
    # parser.add_argument("--reload_path", type=str,
    #                    default='snapshots_2D/final/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_e50.pth')

    ###################################################################################################################################
    # parser.add_argument("--best_epoch", type=int, default=74)
    # parser.add_argument("--best_epoch", type=int, default=51)
    parser.add_argument("--best_epoch", type=int, default=195)
    # parser.add_argument("--best_epoch", type=int, default=176)

    # parser.add_argument("--validsetname", type=str, default='scale')
    parser.add_argument("--validsetname", type=str, default='normal')
    # parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_train_patch_with_white')
    parser.add_argument("--train_list", type=str,
                        default='./data/Mice_Patches/train/data_list.csv')
    # parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    # parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--val_list", type=str,
                        default='./data/Mice_Patches/val/data_list.csv')
    parser.add_argument("--edge_weight", type=float, default=1.2)
    # parser.add_argument("--snapshot_dir", type=str, default='1027results/fold1_with_white_Unet2D_scaleid3_fullydata_1027')
    parser.add_argument("--reload_from_checkpoint",
                        type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='512,512')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    parser.add_argument("--output_folder", type=str, default='output')
    return parser


def count_score_only_two(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki, :, rmin[ki]:rmax[ki], cmin[ki]:cmax[ki]]
        label = labels[ki, :, rmin[ki]:rmax[ki], cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        # preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds1 = pred[1, ...].flatten().detach().cpu().numpy()
        # labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1, ...].detach().flatten().detach().cpu().numpy()

        Val_F1 += f1_score(preds1, labels1, average='macro')

    return Val_F1/cnt, Val_DICE/cnt, 0., 0.


def surfd(input1, input2, sampling=1, connectivity=1):
    # input_1 = np.atleast_1d(input1.astype(bool))
    # input_2 = np.atleast_1d(input2.astype(bool))

    conn = morphology.generate_binary_structure(input1.ndim, connectivity)

    S = input1 - morphology.binary_erosion(input1, conn)
    Sprime = input2 - morphology.binary_erosion(input2, conn)

    S = np.atleast_1d(S.astype(bool))
    Sprime = np.atleast_1d(Sprime.astype(bool))

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return np.max(sds), np.mean(sds)


def count_score(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_HD = 0
    Val_MSD = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki, :, rmin[ki]:rmax[ki], cmin[ki]:cmax[ki]]
        label = labels[ki, :, rmin[ki]:rmax[ki], cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        # preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds0 = pred[1, ...].detach().cpu().numpy()
        labels0 = label[1, ...].detach().detach().cpu().numpy()

        preds1 = pred[1, ...].flatten().detach().cpu().numpy()
        # labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1, ...].detach().flatten().detach().cpu().numpy()

        # try:
        ###############################################################################
        hausdorff, meansurfaceDistance = surfd(preds0, labels0)
        Val_HD += hausdorff
        Val_MSD += meansurfaceDistance
        # 3

        Val_F1 += f1_score(preds1, labels1, average='macro')

        # except:
        #     Val_DICE += 1.
        #     Val_F1 += 1.
        #     Val_HD += 0.
        #     Val_MSD += 0.

    return Val_F1/cnt, Val_DICE/cnt, Val_HD/cnt, Val_MSD/cnt


def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()


def mask_to_box(tensor):
    tensor = tensor.permute([0, 2, 3, 1]).cpu().numpy()
    rmin = np.zeros((4))
    rmax = np.zeros((4))
    cmin = np.zeros((4))
    cmax = np.zeros((4))
    for ki in range(len(tensor)):
        rows = np.any(tensor[ki], axis=1)
        cols = np.any(tensor[ki], axis=0)

        rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
        cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]

    # plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
    return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        criterion = None
        # model = UNet2D_ns(num_classes=args.num_classes, weight_std = False)
        model = UNet2D_scale(num_classes=args.num_classes, weight_std=False)
        check_wo_gpu = 1

        if torch.cuda.is_available() and not check_wo_gpu:
            if args.FP16:
                print("Note: Using FP16 during training************")
                model, optimizer = amp.initialize(
                    model, optimizer, opt_level="O1")
            if args.num_gpus > 1:
                model = engine.data_parallel(model)

        # if torch.cuda.is_available() and not check_wo_gpu:
        #     device = torch.device('cuda:{}'.format(args.local_rank))

        else:
            device = torch.device('cpu')
            print("Running on CPU")
            model.to(device)

        print(
            f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

        if not check_wo_gpu:
            device = torch.device('cpu')
            # device = torch.device('cuda:{}'.format(args.local_rank))
            model.to(device)

        optimizer = torch.optim.SGD(
            model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        if not check_wo_gpu:
            if args.FP16:
                print("Note: Using FP16 during training************")
                model, optimizer = amp.initialize(
                    model, optimizer, opt_level="O1")

            if args.num_gpus > 1:
                model = engine.data_parallel(model)

        print(f'Loading checkpoint from: {args.reload_path}')
        checkpoint = load_model_checkpoint(args.reload_path, device='cpu')
        model.load_state_dict(checkpoint)

        # try:
        #     checkpoint = torch.load(
        #         args.reload_path + '/data.pkl', map_location=torch.device('cpu'), weights_only=False)
        # except:
        #     # Try alternative loading
        #     with open(args.reload_path + '/data.pkl', 'rb') as f:
        #         checkpoint = pickle.load(f)

        # model.load_state_dict(checkpoint)

        # load checkpoint...
        # if args.reload_from_checkpoint:
        #     print('loading from checkpoint: {}'.format(args.reload_path))
        #     if os.path.exists(args.reload_path):
        #         if args.FP16:
        #             checkpoint = torch.load(
        #                 args.reload_path, map_location=torch.device('cpu'), weights_only=False)
        #             model.load_state_dict(checkpoint['model'])
        #             optimizer.load_state_dict(checkpoint['optimizer'])
        #             amp.load_state_dict(checkpoint['amp'])
        #         else:
        #             checkpoint = torch.load(
        #                 args.reload_path + '/data.pkl', map_location=torch.device('cpu'), weights_only=False)
        #             model.load_state_dict(checkpoint)
        #             # model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
        #     else:
        #         print('File not exists in the reload path: {}'.format(
        #             args.reload_path))

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        edge_weight = args.edge_weight

        num_worker = 8

        valloader = DataLoader(
            MOTSValDataSet_joint(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                                 crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                                 edge_weight=edge_weight), batch_size=1, shuffle=False, num_workers=num_worker)

        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999
        batch_size = args.batch_size
        # for epoch in range(0,args.num_epochs):

        # checkpoint = torch.load(
        #     args.reload_path + '/data.pkl', map_location=torch.device('cpu'), weights_only=False)
        # model.load_state_dict(checkpoint)
        # model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))

        model.eval()
        task0_pool_image = ImagePool(8)
        task0_pool_origin = ImagePool(8)
        # task0_pool_mask = ImagePool(8)
        task0_scale = []
        task0_name = []
        # task1_pool_image = ImagePool(8)
        # task1_pool_mask = ImagePool(8)
        # task1_scale = []
        # task1_name = []
        # task2_pool_image = ImagePool(8)
        # task2_pool_mask = ImagePool(8)
        # task2_scale = []
        # task2_name = []
        # task3_pool_image = ImagePool(8)
        # task3_pool_mask = ImagePool(8)
        # task3_scale = []
        # task3_name = []
        # task4_pool_image = ImagePool(8)
        # task4_pool_mask = ImagePool(8)
        # task4_scale = []
        # task4_name = []
        # task5_pool_image = ImagePool(8)
        # task5_pool_mask = ImagePool(8)
        # task5_scale = []
        # task5_name = []
        # ###################################################################
        # task6_pool_image = ImagePool(8)
        # task6_pool_mask = ImagePool(8)
        # task6_scale = []
        # task6_name = []
        #
        # task7_pool_image = ImagePool(8)
        # task7_pool_mask = ImagePool(8)
        # task7_scale = []
        # task7_name = []
        #
        # task8_pool_image = ImagePool(8)
        # task8_pool_mask = ImagePool(8)
        # task8_scale = []
        # task8_name = []
        #
        # task9_pool_image = ImagePool(8)
        # task9_pool_mask = ImagePool(8)
        # task9_scale = []
        # task9_name = []
        #
        # task10_pool_image = ImagePool(8)
        # task10_pool_mask = ImagePool(8)
        # task10_scale = []
        # task10_name = []
        #
        # task11_pool_image = ImagePool(8)
        # task11_pool_mask = ImagePool(8)
        # task11_scale = []
        # task11_name = []
        #
        # task12_pool_image = ImagePool(8)
        # task12_pool_mask = ImagePool(8)
        # task12_scale = []
        # task12_name = []
        #
        # task13_pool_image = ImagePool(8)
        # task13_pool_mask = ImagePool(8)
        # task13_scale = []
        # task13_name = []
        #
        # task14_pool_image = ImagePool(8)
        # task14_pool_mask = ImagePool(8)
        # task14_scale = []
        # task14_name = []
        ###################################################################

        # val_loss = np.zeros((6))
        # val_F1 = np.zeros((6))
        # val_Dice = np.zeros((6))
        # val_HD = np.zeros((6))
        # val_MSD = np.zeros((6))
        # cnt = np.zeros((6))
        ####################################################################
        # val_loss = np.zeros((15))
        # val_F1 = np.zeros((15))
        # val_Dice = np.zeros((15))
        # val_HD = np.zeros((15))
        # val_MSD = np.zeros((15))
        # cnt = np.zeros((15))

        # ####################################################################
        # single_df_0 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_1 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_2 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_3 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_4 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_5 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        #
        # #########################################################################
        # single_df_6 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_7 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_8 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_9 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # ################################################################################
        # single_df_10 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_11 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_12 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_13 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        # single_df_14 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])

        with torch.no_grad():
            for iter, batch in enumerate(valloader):

                'dataloader'
                imgs = batch[0].cpu()
                # lbls = batch[1].cpu()
                # wt = batch[2].cpu().float()
                volumeName = batch[1]
                t_ids = batch[2].cpu()
                s_ids = batch[3]
                backup_shape = batch[4].item()
                origin_images = batch[5].cpu()
                # print(backup_shape)
                # print(origin_images.shape)
                # print(imgs.shape)
                # print(type(backup_shape))
                # print(backup_shape[0])
                # print(type(backup_shape))
                for ki in range(len(imgs)):
                    now_task = t_ids[ki]
                    if now_task == 0:
                        task0_pool_image.add(imgs[ki].unsqueeze(0))

                        # task0_pool_mask.add(lbls[ki].unsqueeze(0))
                        task0_scale.append((s_ids[ki]))
                        task0_name.append((volumeName[ki]))

                        task0_pool_origin.add(origin_images[ki].unsqueeze(0))

                    # elif now_task == 1:
                    #     task1_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task1_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task1_scale.append((s_ids[ki]))
                    #     task1_name.append((volumeName[ki]))
                    # elif now_task == 2:
                    #     task2_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task2_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task2_scale.append((s_ids[ki]))
                    #     task2_name.append((volumeName[ki]))
                    # elif now_task == 3:
                    #     task3_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task3_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task3_scale.append((s_ids[ki]))
                    #     task3_name.append((volumeName[ki]))
                    # elif now_task == 4:
                    #     task4_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task4_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task4_scale.append((s_ids[ki]))
                    #     task4_name.append((volumeName[ki]))
                    # elif now_task == 5:
                    #     task5_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task5_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task5_scale.append((s_ids[ki]))
                    #     task5_name.append((volumeName[ki]))
                    #
                    # #########################################################
                    # elif now_task == 6:
                    #     task6_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task6_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task6_scale.append((s_ids[ki]))
                    #     task6_name.append((volumeName[ki]))
                    #
                    # elif now_task == 7:
                    #     task7_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task7_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task7_scale.append((s_ids[ki]))
                    #     task7_name.append((volumeName[ki]))
                    #
                    # elif now_task == 8:
                    #     task8_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task8_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task8_scale.append((s_ids[ki]))
                    #     task8_name.append((volumeName[ki]))
                    #
                    # elif now_task == 9:
                    #     task9_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task9_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task9_scale.append((s_ids[ki]))
                    #     task9_name.append((volumeName[ki]))
                    # ########################################################
                    # elif now_task == 10:
                    #     task10_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task10_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task10_scale.append((s_ids[ki]))
                    #     task10_name.append((volumeName[ki]))
                    # elif now_task == 11:
                    #     task11_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task11_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task11_scale.append((s_ids[ki]))
                    #     task11_name.append((volumeName[ki]))
                    # elif now_task == 12:
                    #     task12_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task12_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task12_scale.append((s_ids[ki]))
                    #     task12_name.append((volumeName[ki]))
                    # elif now_task == 13:
                    #     task13_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task13_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task13_scale.append((s_ids[ki]))
                    #     task13_name.append((volumeName[ki]))
                    # elif now_task == 14:
                    #     task14_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task14_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task14_scale.append((s_ids[ki]))
                    #     task14_name.append((volumeName[ki]))
                    # 3

                # output_folder = os.path.join(args.snapshot_dir.replace('snapshots_2D/fold1_with_white','/Data/DoDNet/MIDL/MIDL_github/testing_%s' % (args.validsetname)), str(args.best_epoch))

                # For Human
                # output_folder = os.path.join('/data2/DoDNet/MIDL/MIDL_github/Human_out_domain_Omniseg/testing_%s' % (args.validsetname), str(args.best_epoch))
                # For Mice
                # output_folder = os.path.join('/data2/DoDNet/MIDL/MIDL_github/For_Mice/testing_%s' % (args.validsetname),
                #                              str(args.best_epoch))
                # output_folder = os.path.join('/data2/DoDNet/MIDL/MIDL_github/Yue_15/testing_%s' % (args.validsetname),
                #                              str(args.best_epoch))
                # output_folder = os.path.join('/data2/DoDNet/MIDL/MIDL_github/Yue_15/testing_%s' % (args.validsetname))
                # print(os.path.join('output/testing_%s' % (args.validsetname)))
                # output_folder = os.path.join('output/testing_%s' % (args.validsetname))

                output_folder = os.path.join(
                    args.output_folder, 'testing_%s' % (args.validsetname))
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                optimizer.zero_grad()

                # print("task0", task0_pool_image.num_imgs)

                if task0_pool_image.num_imgs >= batch_size:
                    images = task0_pool_image.query(batch_size)
                    # origin = task0_pool_origin.query(batch_size)
                    # print(images.shape)
                    # print(origin.shape)
                    # labels = task0_pool_mask.query(batch_size)

                    scales = torch.ones(batch_size).cpu()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task0_scale.pop(0)
                        filename.append(task0_name.pop(0))

                    preds_0,  _ = model(images, torch.ones(
                        batch_size, device=images.device) * 0,  scales)
                    preds_1,  _ = model(images, torch.ones(
                        batch_size, device=images.device) * 1,  scales)
                    preds_2,  _ = model(images, torch.ones(
                        batch_size, device=images.device) * 2,  scales)
                    preds_3,  _ = model(images, torch.ones(
                        batch_size, device=images.device) * 3,  scales)
                    preds_4,  _ = model(images, torch.ones(
                        batch_size, device=images.device) * 4,  scales)
                    preds_5,  _ = model(images, torch.ones(
                        batch_size, device=images.device) * 5,  scales)
                    preds_6,  _ = model(images, torch.ones(
                        batch_size, device=images.device) * 6,  scales)
                    preds_7,  _ = model(images, torch.ones(
                        batch_size, device=images.device) * 7,  scales)
                    preds_8,  _ = model(images, torch.ones(
                        batch_size, device=images.device) * 8,  scales)
                    preds_9,  _ = model(images, torch.ones(
                        batch_size, device=images.device) * 9,  scales)
                    preds_10, _ = model(images, torch.ones(
                        batch_size, device=images.device) * 10, scales)
                    preds_11, _ = model(images, torch.ones(
                        batch_size, device=images.device) * 11, scales)
                    preds_12, _ = model(images, torch.ones(
                        batch_size, device=images.device) * 12, scales)
                    preds_13, _ = model(images, torch.ones(
                        batch_size, device=images.device) * 13, scales)
                    preds_14, _ = model(images, torch.ones(
                        batch_size, device=images.device) * 14, scales)

                    # preds_0, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 0, scales)
                    # preds_1, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 1, scales)
                    # preds_2, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 2, scales)
                    # preds_3, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 3, scales)
                    # preds_4, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 4, scales)
                    # preds_5, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 5, scales)
                    # preds_6, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 6, scales)
                    # preds_7, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 7, scales)
                    # preds_8, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 8, scales)
                    # preds_9, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 9, scales)
                    # preds_10, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 10, scales)
                    # preds_11, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 11, scales)
                    # preds_12, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 12, scales)
                    # preds_13, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 13, scales)
                    # preds_14, _ = model(images, torch.ones(
                    #     batch_size).cpu() * 14, scales)
                    # now_preds = preds[:,1,...] > preds[:,0,...]
                    # now_preds_onehot = one_hot_2D(now_preds.long())

                    # labels_onehot = one_hot_2D(labels.long())

                    # rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        # prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        out_image = images[pi, ...].permute(
                            [1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / \
                            (out_image.max() - out_image.min())
                        # plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                        #            origin[pi].cpu().numpy())
                        # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                        #           labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)

                        # plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' %(now_task.item())),
                        #            prediction_0.detach().cpu().numpy(), cmap = cm.gray)
                        # print(prediction_0.detach().cpu().numpy().shape)
                        # print(backup_shape)

                        now_task = torch.tensor(0)
                        prediction_0 = preds_0[pi,
                                               1, ...] > preds_0[pi, 0, ...]
                        back_mask_0 = prediction_0.detach().cpu().numpy()

                        now_task = torch.tensor(1)
                        prediction_1 = preds_1[pi,
                                               1, ...] > preds_1[pi, 0, ...]
                        back_mask_1 = prediction_1.detach().cpu().numpy()

                        now_task = torch.tensor(2)
                        prediction_2 = preds_2[pi,
                                               1, ...] > preds_2[pi, 0, ...]
                        back_mask_2 = prediction_2.detach().cpu().numpy()

                        now_task = torch.tensor(3)
                        prediction_3 = preds_3[pi,
                                               1, ...] > preds_3[pi, 0, ...]
                        back_mask_3 = prediction_3.detach().cpu().numpy()

                        now_task = torch.tensor(4)
                        prediction_4 = preds_4[pi,
                                               1, ...] > preds_4[pi, 0, ...]
                        back_mask_4 = prediction_4.detach().cpu().numpy()

                        now_task = torch.tensor(5)
                        prediction_5 = preds_5[pi,
                                               1, ...] > preds_5[pi, 0, ...]
                        back_mask_5 = prediction_5.detach().cpu().numpy()

                        now_task = torch.tensor(6)
                        prediction_6 = preds_6[pi,
                                               1, ...] > preds_6[pi, 0, ...]
                        back_mask_6 = prediction_6.detach().cpu().numpy()

                        now_task = torch.tensor(7)
                        prediction_7 = preds_7[pi,
                                               1, ...] > preds_7[pi, 0, ...]
                        back_mask_7 = prediction_7.detach().cpu().numpy()

                        now_task = torch.tensor(8)
                        prediction_8 = preds_8[pi,
                                               1, ...] > preds_8[pi, 0, ...]
                        back_mask_8 = prediction_8.detach().cpu().numpy()

                        now_task = torch.tensor(9)
                        prediction_9 = preds_9[pi,
                                               1, ...] > preds_9[pi, 0, ...]
                        back_mask_9 = prediction_9.detach().cpu().numpy()

                        now_task = torch.tensor(10)
                        prediction_10 = preds_10[pi,
                                                 1, ...] > preds_10[pi, 0, ...]
                        back_mask_10 = prediction_10.detach().cpu().numpy()

                        now_task = torch.tensor(11)
                        prediction_11 = preds_11[pi,
                                                 1, ...] > preds_11[pi, 0, ...]
                        back_mask_11 = prediction_11.detach().cpu().numpy()

                        now_task = torch.tensor(12)
                        prediction_12 = preds_12[pi,
                                                 1, ...] > preds_12[pi, 0, ...]
                        back_mask_12 = prediction_12.detach().cpu().numpy()

                        now_task = torch.tensor(13)
                        prediction_13 = preds_13[pi,
                                                 1, ...] > preds_13[pi, 0, ...]
                        back_mask_13 = prediction_13.detach().cpu().numpy()

                        now_task = torch.tensor(14)
                        prediction_14 = preds_14[pi,
                                                 1, ...] > preds_14[pi, 0, ...]
                        back_mask_14 = prediction_14.detach().cpu().numpy()

                        print("saving.npy")
                        # print(origin_images[pi].cpu().numpy().shape)

                        # merged_mask = np.concatenate((origin[pi].cpu().numpy(),
                        #                              np.expand_dims(back_mask_0, axis=2), np.expand_dims(back_mask_1, axis=2),
                        #                              np.expand_dims(back_mask_2, axis=2), np.expand_dims(back_mask_3, axis=2),
                        #                              np.expand_dims(back_mask_4, axis=2), np.expand_dims(back_mask_5, axis=2),
                        #                              np.expand_dims(back_mask_6, axis=2), np.expand_dims(back_mask_7, axis=2),
                        #                              np.expand_dims(back_mask_8, axis=2), np.expand_dims(back_mask_9, axis=2),
                        #                              np.expand_dims(back_mask_10, axis=2), np.expand_dims(back_mask_11, axis=2),
                        #                              np.expand_dims(back_mask_12, axis=2), np.expand_dims(back_mask_13, axis=2),
                        #                              np.expand_dims(back_mask_14, axis=2)),
                        #                              axis=2)
                        merged_mask = np.concatenate(
                            (np.expand_dims(back_mask_0, axis=2), np.expand_dims(back_mask_1, axis=2),
                             np.expand_dims(back_mask_2, axis=2), np.expand_dims(
                                 back_mask_3, axis=2),
                             np.expand_dims(back_mask_4, axis=2), np.expand_dims(
                                 back_mask_5, axis=2),
                             np.expand_dims(back_mask_6, axis=2), np.expand_dims(
                                 back_mask_7, axis=2),
                             np.expand_dims(back_mask_8, axis=2), np.expand_dims(
                                 back_mask_9, axis=2),
                             np.expand_dims(back_mask_10, axis=2), np.expand_dims(
                                 back_mask_11, axis=2),
                             np.expand_dims(back_mask_12, axis=2), np.expand_dims(
                                 back_mask_13, axis=2),
                             np.expand_dims(back_mask_14, axis=2)),
                            axis=2)
                        np.save(os.path.join(output_folder, os.path.basename(
                            filename[pi]) + '_preds_merged.npy'), merged_mask)
                        print("Completed saving.npy")
                        # plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_merged.png'),
                        #            merged_mask)
                        # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                        #                                 rmin, rmax, cmin, cmax)
                        # row = len(single_df_0)
                        # single_df_0.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                        #
                        # val_F1[0] += F1
                        # val_Dice[0] += DICE
                        # val_HD[0] += HD
                        # val_MSD[0] += MSD
                        # cnt[0] += 1

            if (task0_pool_image.num_imgs < batch_size) & (task0_pool_image.num_imgs > 0):
                left_size = task0_pool_image.num_imgs
                images = task0_pool_image.query(left_size)
                origin = task0_pool_origin.query(left_size)
                # labels = task0_pool_mask.query(left_size)

                # last_image = torch.unsqueeze(images[-1], dim=0)
                # replicated_image = last_image.repeat(batch_size - left_size, 1, 1, 1)
                # images = torch.cat((images, replicated_image), dim=0)
                #
                # last_label = torch.unsqueeze(labels[-1], dim=0)
                # replicated_label = last_label.repeat(batch_size - left_size, 1, 1)
                # labels = torch.cat((labels, replicated_label), dim=0)

                # now_task = torch.tensor(0)
                scales = torch.ones(left_size).cpu()
                filename = []
                for bi in range(len(scales)):
                    scales[int(len(scales) - 1 - bi)] = task0_scale.pop(0)
                    filename.append(task0_name.pop(0))

                preds_0, _ = model(images, torch.ones(
                    batch_size).cpu() * 0, scales)
                preds_1, _ = model(images, torch.ones(
                    batch_size).cpu() * 1, scales)
                preds_2, _ = model(images, torch.ones(
                    batch_size).cpu() * 2, scales)
                preds_3, _ = model(images, torch.ones(
                    batch_size).cpu() * 3, scales)
                preds_4, _ = model(images, torch.ones(
                    batch_size).cpu() * 4, scales)
                preds_5, _ = model(images, torch.ones(
                    batch_size).cpu() * 5, scales)
                preds_6, _ = model(images, torch.ones(
                    batch_size).cpu() * 6, scales)
                preds_7, _ = model(images, torch.ones(
                    batch_size).cpu() * 7, scales)
                preds_8, _ = model(images, torch.ones(
                    batch_size).cpu() * 8, scales)
                preds_9, _ = model(images, torch.ones(
                    batch_size).cpu() * 9, scales)
                preds_10, _ = model(images, torch.ones(
                    batch_size).cpu() * 10, scales)
                preds_11, _ = model(images, torch.ones(
                    batch_size).cpu() * 11, scales)
                preds_12, _ = model(images, torch.ones(
                    batch_size).cpu() * 12, scales)
                preds_13, _ = model(images, torch.ones(
                    batch_size).cpu() * 13, scales)
                preds_14, _ = model(images, torch.ones(
                    batch_size).cpu() * 14, scales)
                # now_preds = preds[:,1,...] > preds[:,0,...]

                # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                # now_preds_onehot = one_hot_2D(now_preds.long())
                #
                # labels_onehot = one_hot_2D(labels.long())
                #
                # rmin, rmax, cmin, cmax = mask_to_box(images)

                for pi in range(len(images)):
                    # prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                    out_image = images[pi, ...].permute(
                        [1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / \
                        (out_image.max() - out_image.min())
                    # plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                    #            origin[pi].cpu().numpy())
                    # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                    #           labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)

                    # plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' %(now_task.item())),
                    #            prediction_0.detach().cpu().numpy(), cmap = cm.gray)
                    # print(prediction_0.detach().cpu().numpy().shape)
                    # print(backup_shape)

                    now_task = torch.tensor(0)
                    prediction_0 = preds_0[pi, 1, ...] > preds_0[pi, 0, ...]
                    back_mask_0 = prediction_0.detach().cpu().numpy()

                    now_task = torch.tensor(1)
                    prediction_1 = preds_1[pi, 1, ...] > preds_1[pi, 0, ...]
                    back_mask_1 = prediction_1.detach().cpu().numpy()

                    now_task = torch.tensor(2)
                    prediction_2 = preds_2[pi, 1, ...] > preds_2[pi, 0, ...]
                    back_mask_2 = prediction_2.detach().cpu().numpy()

                    now_task = torch.tensor(3)
                    prediction_3 = preds_3[pi, 1, ...] > preds_3[pi, 0, ...]
                    back_mask_3 = prediction_3.detach().cpu().numpy()

                    now_task = torch.tensor(4)
                    prediction_4 = preds_4[pi, 1, ...] > preds_4[pi, 0, ...]
                    back_mask_4 = prediction_4.detach().cpu().numpy()

                    now_task = torch.tensor(5)
                    prediction_5 = preds_5[pi, 1, ...] > preds_5[pi, 0, ...]
                    back_mask_5 = prediction_5.detach().cpu().numpy()

                    now_task = torch.tensor(6)
                    prediction_6 = preds_6[pi, 1, ...] > preds_6[pi, 0, ...]
                    back_mask_6 = prediction_6.detach().cpu().numpy()

                    now_task = torch.tensor(7)
                    prediction_7 = preds_7[pi, 1, ...] > preds_7[pi, 0, ...]
                    back_mask_7 = prediction_7.detach().cpu().numpy()

                    now_task = torch.tensor(8)
                    prediction_8 = preds_8[pi, 1, ...] > preds_8[pi, 0, ...]
                    back_mask_8 = prediction_8.detach().cpu().numpy()

                    now_task = torch.tensor(9)
                    prediction_9 = preds_9[pi, 1, ...] > preds_9[pi, 0, ...]
                    back_mask_9 = prediction_9.detach().cpu().numpy()

                    now_task = torch.tensor(10)
                    prediction_10 = preds_10[pi, 1, ...] > preds_10[pi, 0, ...]
                    back_mask_10 = prediction_10.detach().cpu().numpy()

                    now_task = torch.tensor(11)
                    prediction_11 = preds_11[pi, 1, ...] > preds_11[pi, 0, ...]
                    back_mask_11 = prediction_11.detach().cpu().numpy()

                    now_task = torch.tensor(12)
                    prediction_12 = preds_12[pi, 1, ...] > preds_12[pi, 0, ...]
                    back_mask_12 = prediction_12.detach().cpu().numpy()

                    now_task = torch.tensor(13)
                    prediction_13 = preds_13[pi, 1, ...] > preds_13[pi, 0, ...]
                    back_mask_13 = prediction_13.detach().cpu().numpy()

                    now_task = torch.tensor(14)
                    prediction_14 = preds_14[pi, 1, ...] > preds_14[pi, 0, ...]
                    back_mask_14 = prediction_14.detach().cpu().numpy()

                    print("saving.npy")
                    # print(origin_images[pi].cpu().numpy().shape)

                    # merged_mask = np.concatenate((origin[pi].cpu().numpy(),
                    #                              np.expand_dims(back_mask_0, axis=2), np.expand_dims(back_mask_1, axis=2),
                    #                              np.expand_dims(back_mask_2, axis=2), np.expand_dims(back_mask_3, axis=2),
                    #                              np.expand_dims(back_mask_4, axis=2), np.expand_dims(back_mask_5, axis=2),
                    #                              np.expand_dims(back_mask_6, axis=2), np.expand_dims(back_mask_7, axis=2),
                    #                              np.expand_dims(back_mask_8, axis=2), np.expand_dims(back_mask_9, axis=2),
                    #                              np.expand_dims(back_mask_10, axis=2), np.expand_dims(back_mask_11, axis=2),
                    #                              np.expand_dims(back_mask_12, axis=2), np.expand_dims(back_mask_13, axis=2),
                    #                              np.expand_dims(back_mask_14, axis=2)),
                    #                              axis=2)
                    merged_mask = np.concatenate(
                        (np.expand_dims(back_mask_0, axis=2), np.expand_dims(back_mask_1, axis=2),
                         np.expand_dims(back_mask_2, axis=2), np.expand_dims(
                             back_mask_3, axis=2),
                         np.expand_dims(back_mask_4, axis=2), np.expand_dims(
                             back_mask_5, axis=2),
                         np.expand_dims(back_mask_6, axis=2), np.expand_dims(
                             back_mask_7, axis=2),
                         np.expand_dims(back_mask_8, axis=2), np.expand_dims(
                             back_mask_9, axis=2),
                         np.expand_dims(back_mask_10, axis=2), np.expand_dims(
                             back_mask_11, axis=2),
                         np.expand_dims(back_mask_12, axis=2), np.expand_dims(
                             back_mask_13, axis=2),
                         np.expand_dims(back_mask_14, axis=2)),
                        axis=2)
                    np.save(os.path.join(output_folder, os.path.basename(
                        filename[pi]) + '_preds_merged.npy'), merged_mask)
                    print("Completed saving.npy")
                    # plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_merged.png'),
                    #            merged_mask)
                    # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                    #                                 rmin, rmax, cmin, cmax)
                    # row = len(single_df_0)
                    # single_df_0.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
                    #
                    # val_F1[0] += F1
                    # val_Dice[0] += DICE
                    # val_HD[0] += HD
                    # val_MSD[0] += MSD
                    # cnt[0] += 1

            # if (task1_pool_image.num_imgs < batch_size) & (task1_pool_image.num_imgs >0):
            #     left_size = task1_pool_image.num_imgs
            #     images = task1_pool_image.query(left_size)
            #     # labels = task1_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task1_scale.pop(0)
            #         filename.append(task1_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 1, scales)
            #     now_task = torch.tensor(1)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     # labels_onehot = one_hot_2D(labels.long())
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_1)
            #         # single_df_1.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[1] += F1
            #         # val_Dice[1] += DICE
            #         # val_HD[1] += HD
            #         # val_MSD[1] += MSD
            #         # cnt[1] += 1
            #
            # if (task2_pool_image.num_imgs < batch_size) & (task2_pool_image.num_imgs >0):
            #     left_size = task2_pool_image.num_imgs
            #     images = task2_pool_image.query(left_size)
            #     # labels = task2_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task2_scale.pop(0)
            #         filename.append(task2_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 2, scales)
            #     now_task = torch.tensor(2)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_2)
            #         # single_df_2.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[2] += F1
            #         # val_Dice[2] += DICE
            #         # val_HD[2] += HD
            #         # val_MSD[2] += MSD
            #         # cnt[2] += 1
            #
            # if (task3_pool_image.num_imgs < batch_size) & (task3_pool_image.num_imgs >0):
            #     left_size = task3_pool_image.num_imgs
            #     images = task3_pool_image.query(left_size)
            #     # labels = task3_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task3_scale.pop(0)
            #         filename.append(task3_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 3, scales)
            #     now_task = torch.tensor(3)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_3)
            #         # single_df_3.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[3] += F1
            #         # val_Dice[3] += DICE
            #         # val_HD[3] += HD
            #         # val_MSD[3] += MSD
            #         # cnt[3] += 1
            #
            # if (task4_pool_image.num_imgs < batch_size) & (task4_pool_image.num_imgs >0):
            #     left_size = task4_pool_image.num_imgs
            #     images = task4_pool_image.query(left_size)
            #     # labels = task4_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task4_scale.pop(0)
            #         filename.append(task4_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 4, scales)
            #     now_task = torch.tensor(4)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_4)
            #         # single_df_4.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[4] += F1
            #         # val_Dice[4] += DICE
            #         # val_HD[4] += HD
            #         # val_MSD[4] += MSD
            #         # cnt[4] += 1
            #
            # if (task5_pool_image.num_imgs < batch_size) & (task5_pool_image.num_imgs >0):
            #     left_size = task5_pool_image.num_imgs
            #     images = task5_pool_image.query(left_size)
            #     # labels = task5_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task5_scale.pop(0)
            #         filename.append(task5_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 5, scales)
            #     now_task = torch.tensor(5)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         num = len(glob.glob(os.path.join(output_folder, '*')))
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_5)
            #         # single_df_5.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[5] += F1
            #         # val_Dice[5] += DICE
            #         # val_HD[5] += HD
            #         # val_MSD[5] += MSD
            #         # cnt[5] += 1
            #
            # if (task6_pool_image.num_imgs < batch_size) & (task6_pool_image.num_imgs >0):
            #     left_size = task6_pool_image.num_imgs
            #     images = task6_pool_image.query(left_size)
            #     # labels = task6_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task6_scale.pop(0)
            #         filename.append(task6_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 6, scales)
            #     now_task = torch.tensor(6)
            #     # print("PREDS.shape:", preds.shape)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         num = len(glob.glob(os.path.join(output_folder, '*')))
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         #print("preds.shape:", now_preds_onehot[pi].unsqueeze(0).shape)
            #         #print("targets.shape:", labels_onehot[pi].unsqueeze(0).shape)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
            #         #                                 labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_6)
            #         # single_df_6.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[6] += F1
            #         # val_Dice[6] += DICE
            #         # val_HD[6] += HD
            #         # val_MSD[6] += MSD
            #         # cnt[6] += 1
            #
            # if (task7_pool_image.num_imgs < batch_size) & (task7_pool_image.num_imgs >0):
            #     left_size = task7_pool_image.num_imgs
            #     images = task7_pool_image.query(left_size)
            #     # labels = task7_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task7_scale.pop(0)
            #         filename.append(task7_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 7, scales)
            #     now_task = torch.tensor(7)
            #     # print("PREDS.shape:", preds.shape)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         num = len(glob.glob(os.path.join(output_folder, '*')))
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         #print("preds.shape:", now_preds_onehot[pi].unsqueeze(0).shape)
            #         #print("targets.shape:", labels_onehot[pi].unsqueeze(0).shape)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
            #         #                                 labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_7)
            #         # single_df_7.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[7] += F1
            #         # val_Dice[7] += DICE
            #         # val_HD[7] += HD
            #         # val_MSD[7] += MSD
            #         # cnt[7] += 1
            #
            # if (task8_pool_image.num_imgs < batch_size) & (task8_pool_image.num_imgs >0):
            #     left_size = task8_pool_image.num_imgs
            #     images = task8_pool_image.query(left_size)
            #     # labels = task8_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task8_scale.pop(0)
            #         filename.append(task8_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 8, scales)
            #     now_task = torch.tensor(8)
            #     # print("PREDS.shape:", preds.shape)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         num = len(glob.glob(os.path.join(output_folder, '*')))
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         #print("preds.shape:", now_preds_onehot[pi].unsqueeze(0).shape)
            #         #print("targets.shape:", labels_onehot[pi].unsqueeze(0).shape)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
            #         #                                 labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_8)
            #         # single_df_8.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[8] += F1
            #         # val_Dice[8] += DICE
            #         # val_HD[8] += HD
            #         # val_MSD[8] += MSD
            #         # cnt[8] += 1
            #
            # if (task9_pool_image.num_imgs < batch_size) & (task9_pool_image.num_imgs >0):
            #     left_size = task9_pool_image.num_imgs
            #     images = task9_pool_image.query(left_size)
            #     # labels = task9_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task9_scale.pop(0)
            #         filename.append(task9_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 9, scales)
            #     now_task = torch.tensor(9)
            #     # print("PREDS.shape:", preds.shape)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         num = len(glob.glob(os.path.join(output_folder, '*')))
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         #print("preds.shape:", now_preds_onehot[pi].unsqueeze(0).shape)
            #         #print("targets.shape:", labels_onehot[pi].unsqueeze(0).shape)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
            #         #                                 labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_9)
            #         # single_df_9.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[9] += F1
            #         # val_Dice[9] += DICE
            #         # val_HD[9] += HD
            #         # val_MSD[9] += MSD
            #         # cnt[9] += 1
            #
            # if (task10_pool_image.num_imgs < batch_size) & (task10_pool_image.num_imgs >0):
            #     left_size = task10_pool_image.num_imgs
            #     images = task10_pool_image.query(left_size)
            #     # labels = task10_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task10_scale.pop(0)
            #         filename.append(task10_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 10, scales)
            #     now_task = torch.tensor(10)
            #     # print("PREDS.shape:", preds.shape)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         num = len(glob.glob(os.path.join(output_folder, '*')))
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         #print("preds.shape:", now_preds_onehot[pi].unsqueeze(0).shape)
            #         #print("targets.shape:", labels_onehot[pi].unsqueeze(0).shape)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
            #         #                                 labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_10)
            #         # single_df_10.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[10] += F1
            #         # val_Dice[10] += DICE
            #         # val_HD[10] += HD
            #         # val_MSD[10] += MSD
            #         # cnt[10] += 1
            #
            # if (task11_pool_image.num_imgs < batch_size) & (task11_pool_image.num_imgs >0):
            #     left_size = task11_pool_image.num_imgs
            #     images = task11_pool_image.query(left_size)
            #     # labels = task11_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task11_scale.pop(0)
            #         filename.append(task11_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 11, scales)
            #     now_task = torch.tensor(11)
            #     # print("PREDS.shape:", preds.shape)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         num = len(glob.glob(os.path.join(output_folder, '*')))
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         #print("preds.shape:", now_preds_onehot[pi].unsqueeze(0).shape)
            #         #print("targets.shape:", labels_onehot[pi].unsqueeze(0).shape)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
            #         #                                 labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_11)
            #         # single_df_11.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[11] += F1
            #         # val_Dice[11] += DICE
            #         # val_HD[11] += HD
            #         # val_MSD[11] += MSD
            #         # cnt[11] += 1
            #
            # if (task12_pool_image.num_imgs < batch_size) & (task12_pool_image.num_imgs >0):
            #     left_size = task12_pool_image.num_imgs
            #     images = task12_pool_image.query(left_size)
            #     # labels = task12_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task12_scale.pop(0)
            #         filename.append(task12_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 12, scales)
            #     now_task = torch.tensor(12)
            #     # print("PREDS.shape:", preds.shape)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         num = len(glob.glob(os.path.join(output_folder, '*')))
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         #print("preds.shape:", now_preds_onehot[pi].unsqueeze(0).shape)
            #         #print("targets.shape:", labels_onehot[pi].unsqueeze(0).shape)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
            #         #                                 labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_12)
            #         # single_df_12.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[12] += F1
            #         # val_Dice[12] += DICE
            #         # val_HD[12] += HD
            #         # val_MSD[12] += MSD
            #         # cnt[12] += 1
            #
            # if (task13_pool_image.num_imgs < batch_size) & (task13_pool_image.num_imgs >0):
            #     left_size = task13_pool_image.num_imgs
            #     images = task13_pool_image.query(left_size)
            #     # labels = task13_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task13_scale.pop(0)
            #         filename.append(task13_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 13, scales)
            #     now_task = torch.tensor(13)
            #     # print("PREDS.shape:", preds.shape)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         num = len(glob.glob(os.path.join(output_folder, '*')))
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         #print("preds.shape:", now_preds_onehot[pi].unsqueeze(0).shape)
            #         #print("targets.shape:", labels_onehot[pi].unsqueeze(0).shape)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
            #         #                                 labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_13)
            #         # single_df_13.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[13] += F1
            #         # val_Dice[13] += DICE
            #         # val_HD[13] += HD
            #         # val_MSD[13] += MSD
            #         # cnt[13] += 1
            #
            # if (task14_pool_image.num_imgs < batch_size) & (task14_pool_image.num_imgs >0):
            #     left_size = task14_pool_image.num_imgs
            #     images = task14_pool_image.query(left_size)
            #     # labels = task14_pool_mask.query(left_size)
            #     scales = torch.ones(left_size).cpu()
            #     filename = []
            #     for bi in range(len(scales)):
            #         scales[int(len(scales) - 1 - bi)] = task14_scale.pop(0)
            #         filename.append(task14_name.pop(0))
            #
            #     preds, _ = model(images, torch.ones(left_size).cpu() * 14, scales)
            #     now_task = torch.tensor(14)
            #     # print("PREDS.shape:", preds.shape)
            #
            #     # now_preds = preds[:, 1, ...] > preds[:, 0, ...]
            #     # now_preds_onehot = one_hot_2D(now_preds.long())
            #     #
            #     # labels_onehot = one_hot_2D(labels.long())
            #     #
            #     # rmin, rmax, cmin, cmax = mask_to_box(images)
            #
            #     for pi in range(len(images)):
            #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
            #         num = len(glob.glob(os.path.join(output_folder, '*')))
            #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
            #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
            #                    img)
            #         # plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
            #         #            labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
            #         plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
            #                    prediction.detach().cpu().numpy(), cmap=cm.gray)
            #
            #         #print("preds.shape:", now_preds_onehot[pi].unsqueeze(0).shape)
            #         #print("targets.shape:", labels_onehot[pi].unsqueeze(0).shape)
            #
            #         # F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
            #         #                                 labels_onehot[pi].unsqueeze(0),
            #         #                                 rmin, rmax, cmin, cmax)
            #         # row = len(single_df_14)
            #         # single_df_14.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]
            #         #
            #         # val_F1[14] += F1
            #         # val_Dice[14] += DICE
            #         # val_HD[14] += HD
            #         # val_MSD[14] += MSD
            #         # cnt[14] += 1

        # avg_val_F1 = val_F1 / cnt
        # avg_val_Dice = val_Dice / cnt
        # avg_val_HD = val_HD / cnt
        # avg_val_MSD = val_MSD / cnt

        # print('Validate \n 0dt_f1={:.4} 0dt_dsc={:.4} 0dt_hd={:.4} 0dt_msd={:.4}'
        #       ' \n 1pt_f1={:.4} 1pt_dsc={:.4} 1pt_hd={:.4} 1pt_msd={:.4}\n'
        #       ' \n 2cps_f1={:.4} 2cps_dsc={:.4} 2cps_hd={:.4} 2cps_msd={:.4}\n'
        #       ' \n 3tf_f1={:.4} 3tf_dsc={:.4} 3tf_hd={:.4} 3tf_msd={:.4}\n'
        #       ' \n 4vs_f1={:.4} 4vs_dsc={:.4} 4vs_hd={:.4} 4vs_msd={:.4}\n'
        #       ' \n 5ptc_f1={:.4} 5ptc_dsc={:.4} 5ptc_hd={:.4} 5ptc_msd={:.4}\n'
        #       ########################################################################
        #       ' \n 6ptc_f1={:.4} 6ptc_dsc={:.4} 6ptc_hd={:.4} 6ptc_msd={:.4}\n'
        #       ' \n 7ptc_f1={:.4} 7ptc_dsc={:.4} 7ptc_hd={:.4} 7ptc_msd={:.4}\n'
        #       ' \n 8ptc_f1={:.4} 8ptc_dsc={:.4} 8ptc_hd={:.4} 8ptc_msd={:.4}\n'
        #       ' \n 9ptc_f1={:.4} 9ptc_dsc={:.4} 9ptc_hd={:.4} 9ptc_msd={:.4}\n'
        #       ' \n 10ptc_f1={:.4} 10ptc_dsc={:.4} 10ptc_hd={:.4} 10ptc_msd={:.4}\n'
        #       ' \n 11ptc_f1={:.4} 11ptc_dsc={:.4} 11ptc_hd={:.4} 11ptc_msd={:.4}\n'
        #       ' \n 12ptc_f1={:.4} 12ptc_dsc={:.4} 12ptc_hd={:.4} 12ptc_msd={:.4}\n'
        #       ' \n 13ptc_f1={:.4} 13ptc_dsc={:.4} 13ptc_hd={:.4} 13ptc_msd={:.4}\n'
        #       ' \n 14ptc_f1={:.4} 14ptc_dsc={:.4} 14ptc_hd={:.4} 14ptc_msd={:.4}\n'
        #       #######################################################################
        #       .format(avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item(),
        #               avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(), avg_val_MSD[1].item(),
        #               avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_HD[2].item(), avg_val_MSD[2].item(),
        #               avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_HD[3].item(), avg_val_MSD[3].item(),
        #               avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_HD[4].item(), avg_val_MSD[4].item(),
        #               avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_HD[5].item(), avg_val_MSD[5].item()
        #               #############################################################################################
        #               , avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_HD[6].item(), avg_val_MSD[6].item()
        #               , avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_HD[7].item(), avg_val_MSD[7].item()
        #               , avg_val_F1[8].item(), avg_val_Dice[8].item(), avg_val_HD[8].item(), avg_val_MSD[8].item()
        #               , avg_val_F1[9].item(), avg_val_Dice[9].item(), avg_val_HD[9].item(), avg_val_MSD[9].item()
        #               #############################################################################################
        #               , avg_val_F1[10].item(), avg_val_Dice[10].item(), avg_val_HD[10].item(), avg_val_MSD[10].item()
        #               , avg_val_F1[11].item(), avg_val_Dice[11].item(), avg_val_HD[11].item(), avg_val_MSD[11].item()
        #               , avg_val_F1[12].item(), avg_val_Dice[12].item(), avg_val_HD[12].item(), avg_val_MSD[12].item()
        #               , avg_val_F1[13].item(), avg_val_Dice[13].item(), avg_val_HD[13].item(), avg_val_MSD[13].item()
        #               , avg_val_F1[14].item(), avg_val_Dice[14].item(), avg_val_HD[14].item(), avg_val_MSD[14].item()
        #               ))
        #
        # df = pd.DataFrame(columns = ['task','F1','Dice','HD','MSD'])
        # df.loc[0] = ['0Ahn', avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item()]
        # df.loc[1] = ['1Cap', avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(), avg_val_MSD[1].item()]
        # df.loc[2] = ['2Glos', avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_HD[2].item(), avg_val_MSD[2].item()]
        # df.loc[3] = ['3Hayc', avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_HD[3].item(), avg_val_MSD[3].item()]
        # df.loc[4] = ['4Hays', avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_HD[4].item(), avg_val_MSD[4].item()]
        # df.loc[5] = ['5meex', avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_HD[5].item(), avg_val_MSD[5].item()]
        # ##########################################################################
        # df.loc[6] = ['6mely', avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_HD[6].item(), avg_val_MSD[6].item()]
        # df.loc[7] = ['7micro', avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_HD[7].item(), avg_val_MSD[7].item()]
        # df.loc[8] = ['8nos', avg_val_F1[8].item(), avg_val_Dice[8].item(), avg_val_HD[8].item(), avg_val_MSD[8].item()]
        # df.loc[9] = ['9segs', avg_val_F1[9].item(), avg_val_Dice[9].item(), avg_val_HD[9].item(), avg_val_MSD[9].item()]
        # ##########################################################################
        # df.loc[10] = ['9segs', avg_val_F1[10].item(), avg_val_Dice[10].item(), avg_val_HD[10].item(), avg_val_MSD[10].item()]
        # df.loc[11] = ['9segs', avg_val_F1[11].item(), avg_val_Dice[11].item(), avg_val_HD[11].item(), avg_val_MSD[11].item()]
        # df.loc[12] = ['9segs', avg_val_F1[12].item(), avg_val_Dice[12].item(), avg_val_HD[12].item(), avg_val_MSD[12].item()]
        # df.loc[13] = ['9segs', avg_val_F1[13].item(), avg_val_Dice[13].item(), avg_val_HD[13].item(), avg_val_MSD[13].item()]
        # df.loc[14] = ['9segs', avg_val_F1[14].item(), avg_val_Dice[14].item(), avg_val_HD[14].item(), avg_val_MSD[14].item()]
        # ##########################################################################
        # df.to_csv(os.path.join(output_folder,'testing_result.csv'))
        #
        # single_df_0.to_csv(os.path.join(output_folder,'testing_result_0.csv'))
        # single_df_1.to_csv(os.path.join(output_folder,'testing_result_1.csv'))
        # single_df_2.to_csv(os.path.join(output_folder,'testing_result_2.csv'))
        # single_df_3.to_csv(os.path.join(output_folder,'testing_result_3.csv'))
        # single_df_4.to_csv(os.path.join(output_folder,'testing_result_4.csv'))
        # single_df_5.to_csv(os.path.join(output_folder,'testing_result_5.csv'))
        # ###############################################################################
        # single_df_6.to_csv(os.path.join(output_folder, 'testing_result_6.csv'))
        # single_df_7.to_csv(os.path.join(output_folder, 'testing_result_7.csv'))
        # single_df_8.to_csv(os.path.join(output_folder, 'testing_result_8.csv'))
        # single_df_9.to_csv(os.path.join(output_folder, 'testing_result_9.csv'))
        # ###############################################################################
        # single_df_10.to_csv(os.path.join(output_folder, 'testing_result_10.csv'))
        # single_df_11.to_csv(os.path.join(output_folder, 'testing_result_11.csv'))
        # single_df_12.to_csv(os.path.join(output_folder, 'testing_result_12.csv'))
        # single_df_13.to_csv(os.path.join(output_folder, 'testing_result_13.csv'))
        # single_df_14.to_csv(os.path.join(output_folder, 'testing_result_14.csv'))
        # ###############################################################################

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
