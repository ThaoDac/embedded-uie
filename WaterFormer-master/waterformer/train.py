import argparse
import datetime
import logging
import math
import random
import time
import torch
import os
import sys
from os import path as osp

# Add waterformer directory to Python path
waterformer_dir = osp.dirname(osp.abspath(__file__))
if waterformer_dir not in sys.path:
    sys.path.insert(0, waterformer_dir)

from data import create_dataloader, create_dataset
from data.data_sampler import EnlargedSampler
from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from models import create_model
from utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from utils.dist_util import get_dist_info, init_dist
from utils.options import dict2str, parse

import numpy as np


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--config', type=str, default=None, help='Path to unified config YAML file (optional, overrides opt params)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # Load unified config if provided (takes precedence over opt file)
    if args.config is not None and os.path.exists(args.config):
        print(f'\n✓ Loading unified config from: {args.config}')
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            unified_cfg = yaml.safe_load(f)

        # Override dataset paths
        if 'dataset' in unified_cfg:
            if 'train_input' in unified_cfg['dataset'] and 'train_gt' in unified_cfg['dataset']:
                opt['datasets']['train']['dataroot_lq'] = unified_cfg['dataset']['train_input']
                opt['datasets']['train']['dataroot_gt'] = unified_cfg['dataset']['train_gt']
                print(f"  Config override: train dataroot_lq = {opt['datasets']['train']['dataroot_lq']}")
                print(f"  Config override: train dataroot_gt = {opt['datasets']['train']['dataroot_gt']}")

            if 'test_input' in unified_cfg['dataset'] and 'test_gt' in unified_cfg['dataset']:
                if 'val' in opt['datasets']:
                    opt['datasets']['val']['dataroot_lq'] = unified_cfg['dataset']['test_input']
                    opt['datasets']['val']['dataroot_gt'] = unified_cfg['dataset']['test_gt']
                    print(f"  Config override: val dataroot_lq = {opt['datasets']['val']['dataroot_lq']}")
                    print(f"  Config override: val dataroot_gt = {opt['datasets']['val']['dataroot_gt']}")

            if 'img_size' in unified_cfg['dataset']:
                img_size = unified_cfg['dataset']['img_size']
                opt['datasets']['train']['gt_size'] = img_size
                # Update progressive training sizes proportionally
                original_max = max(opt['datasets']['train'].get('gt_sizes', [384]))
                scale_factor = img_size / original_max
                if 'gt_sizes' in opt['datasets']['train']:
                    opt['datasets']['train']['gt_sizes'] = [
                        int(size * scale_factor) for size in opt['datasets']['train']['gt_sizes']
                    ]
                print(f"  Config override: gt_size = {img_size}")

        # Override training parameters
        if 'training' in unified_cfg:
            if 'epochs' in unified_cfg['training']:
                # Convert epochs to iterations (approximate)
                # Assuming average dataset size, will be recalculated later
                epochs = unified_cfg['training']['epochs']
                # Keep original total_iter for now, will scale based on actual dataset
                opt['train']['total_iter_target_epochs'] = epochs
                print(f"  Config override: target_epochs = {epochs}")

            if 'batch_size' in unified_cfg['training']:
                batch_size = unified_cfg['training']['batch_size']
                opt['datasets']['train']['batch_size_per_gpu'] = batch_size
                # Update mini_batch_sizes for progressive training
                if 'mini_batch_sizes' in opt['datasets']['train']:
                    # Scale down all batch sizes proportionally
                    opt['datasets']['train']['mini_batch_sizes'] = [batch_size] * len(opt['datasets']['train']['mini_batch_sizes'])
                print(f"  Config override: batch_size_per_gpu = {batch_size}")

            if 'lr' in unified_cfg['training']:
                lr = unified_cfg['training']['lr']
                if 'optim_g' in opt['train']:
                    opt['train']['optim_g']['lr'] = lr
                print(f"  Config override: lr = {lr}")

            if 'num_workers' in unified_cfg['training']:
                num_workers = unified_cfg['training']['num_workers']
                opt['datasets']['train']['num_worker_per_gpu'] = num_workers
                if 'val' in opt['datasets']:
                    opt['datasets']['val']['num_worker_per_gpu'] = num_workers
                print(f"  Config override: num_worker_per_gpu = {num_workers}")

            if 'seed' in unified_cfg['training']:
                seed = unified_cfg['training']['seed']
                opt['manual_seed'] = seed
                print(f"  Config override: manual_seed = {seed}")

            if 'gradient_accumulation_steps' in unified_cfg['training']:
                grad_accum = unified_cfg['training']['gradient_accumulation_steps']
                opt['train']['gradient_accumulation_steps'] = grad_accum
                print(f"  Config override: gradient_accumulation_steps = {grad_accum}")
                print(f"  Effective batch size = batch_size ({batch_size}) × grad_accum ({grad_accum}) = {batch_size * grad_accum}")

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='waterformer', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))

            # If unified config specified target epochs, recalculate total_iter
            if 'total_iter_target_epochs' in opt['train']:
                target_epochs = opt['train']['total_iter_target_epochs']
                total_iters = int(target_epochs * num_iter_per_epoch)
                opt['train']['total_iter'] = total_iters
                logger.info(f'Recalculated total_iter from target_epochs={target_epochs}: {total_iters} iterations')
            else:
                total_iters = int(opt['train']['total_iter'])

            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'work_dirs/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # Best model tracking
    best_metric = 0.0  # Track best validation metric (higher is better for PSNR/SSIM)
    best_iter = 0
    logger.info('Best model will be saved based on validation metric (higher is better)')

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            
            ### ------Progressive learning ---------------------
            j = ((current_iter>groups) !=True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]
            
            if logger_j[bs_j]:
                # logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size, mini_batch_size*torch.cuda.device_count()))
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size, mini_batch_size))
                logger_j[bs_j] = False

            lq = train_data['lq']
            gt = train_data['gt']

            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:,:,x0:x1,y0:y1]
                gt = gt[:,:,x0*scale:x1*scale,y0*scale:y1*scale]
            ###-------------------------------------------

            
            model.feed_train_data({'lq': lq, 'gt':gt})
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                # Clear CUDA cache before validation to free up memory
                torch.cuda.empty_cache()

                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                current_metric = model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image)

                # Clear CUDA cache after validation
                torch.cuda.empty_cache()

                # Save best model if current metric is better
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_iter = current_iter
                    logger.info(f'Saving best model at iter {current_iter} with metric: {best_metric:.4f}')
                    model.save_network(model.net_g, 'net_g', 'best')
                    # Also save training state for best model
                    import shutil
                    latest_state = osp.join(opt['path']['training_states'], f'{current_iter}.state')
                    best_state = osp.join(opt['path']['training_states'], 'best.state')
                    if osp.exists(latest_state):
                        shutil.copy(latest_state, best_state)

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info(f'Best model was saved at iter {best_iter} with metric: {best_metric:.4f}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
