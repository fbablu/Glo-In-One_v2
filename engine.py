import os
import os.path as osp
import time
import argparse

import torch
import torch.distributed as dist

from utils_engine.logger import get_logger
from utils_engine.pyt_utils import all_reduce_tensor, extant_file

# Handle apex imports with fallback
try:
    from apex.parallel import DistributedDataParallel as ApexDDP, SyncBatchNorm
    APEX_AVAILABLE = True
except ImportError:
    print("Warning: NVIDIA apex not found. Using PyTorch native alternatives.")
    ApexDDP = None
    SyncBatchNorm = None
    APEX_AVAILABLE = False

logger = get_logger()


class Engine(object):
    def __init__(self, custom_parser=None):
        logger.info(
            "PyTorch Version {}".format(torch.__version__))
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath

        # if not self.args.gpu == 'None':
        #     os.environ["CUDA_VISIBLE_DEVICES"]=self.args.gpu

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1
            print("WORLD_SIZE is %d" % (int(os.environ['WORLD_SIZE'])))
        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
            self.devices = [i for i in range(self.world_size)]
        else:
            gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            self.devices = [i for i in range(len(gpus.split(',')))]

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        # p.add_argument('--local_rank', default=0, type=int,
        #                help='process rank on node')

    def data_parallel(self, model):
        if self.distributed:
            # Use Apex DDP if available, otherwise fall back to PyTorch native DDP
            if APEX_AVAILABLE and ApexDDP is not None:
                model = ApexDDP(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[
                        self.local_rank], output_device=self.local_rank
                )
        else:
            model = torch.nn.DataParallel(model)
        return model

    def get_train_loader(self, train_dataset, collate_fn=None):
        train_sampler = None
        is_shuffle = True
        batch_size = self.args.batch_size

        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
            batch_size = self.args.batch_size // self.world_size
            is_shuffle = False

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=self.args.num_workers,
                                                   drop_last=False,
                                                   shuffle=is_shuffle,
                                                   pin_memory=True,
                                                   sampler=train_sampler,
                                                   collate_fn=collate_fn)

        return train_loader, train_sampler

    def get_test_loader(self, test_dataset):
        test_sampler = None
        is_shuffle = False
        batch_size = self.args.batch_size

        if self.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset)
            batch_size = self.args.batch_size // self.world_size

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  num_workers=self.args.num_workers,
                                                  drop_last=False,
                                                  shuffle=is_shuffle,
                                                  pin_memory=True,
                                                  sampler=test_sampler)

        return test_loader, test_sampler

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return all_reduce_tensor(tensor, world_size=self.world_size, norm=norm)
        else:
            return torch.mean(tensor)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False
