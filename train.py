import torch
import inspect
import json
import yaml
import math
import os
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import diceloss
import pandas as pd
from thop import profile

import numpy as np
from functools import partial
from os.path import expanduser, join, isfile, basename

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext
from torch.utils.data import DataLoader

from general_utils import TrainingLogger, get_attribute, filter_args, log, training_config_from_cli_args
import random


def cosine_warmup_lr(i, warmup=10, max_iter=90):
    """ Cosine LR with Warmup """
    if i < warmup:
        return (i + 1) / (warmup + 1)
    else:
        return 0.5 + 0.5 * math.cos(math.pi * (((i - warmup) / (max_iter - warmup))))


def validate(model, dataset, config):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    metric_class, use_metric = config.val_metric_class, config.use_val_metric
    Dice_loss = diceloss.SoftDiceLoss()


    model.eval()
    model.cuda()

    if metric_class is not None:
        metric = get_attribute(metric_class)()

    with ((torch.no_grad())):

        i, losses = 0, []
        for data_x, data_y in data_loader:

            data_x = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_x]
            data_y = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_y]

            prompts = model.sample_prompts(data_x[1], prompt_list=('a photo of a {}',))
            pred = model(data_x[0], prompts, return_features=False)

            if metric_class is not None:
                metric.add([pred], data_y)

            loss = Dice_loss(pred, data_y[0].cuda())
            losses += [float(loss)]

            i += 1

            if config.val_max_iterations is not None and i > config.val_max_iterations:
                break

    if use_metric is None:
        return np.mean(losses), {}, False
    else:
        metric_scores = {m: s for m, s in zip(metric.names(), metric.value())} if metric is not None else {}
        return np.mean(losses), metric_scores, True


def save_loss_plot(train_losses, val_losses, val_interval, filename):
    """ Save and update the training and validation loss """
    plt.figure()
    plt.plot(train_losses, label='Training Loss')

    if val_losses:
        val_x = list(range(val_interval - 1, val_interval * len(val_losses), val_interval))
        plt.plot(val_x, val_losses, label='Validation Loss', linestyle='--')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()



def main():
    config = training_config_from_cli_args()

    val_interval, best_val_loss, best_val_score = config.val_interval, float('inf'), float('-inf')


    model_cls = get_attribute(config.model)
    _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)
    model = model_cls(**model_args).cuda()

    # Instantiate dataset
    dataset_cls = get_attribute(config.dataset)
    _, dataset_args, _ = filter_args(config, inspect.signature(dataset_cls).parameters)
    dataset = dataset_cls(**dataset_args)

    log.info(f'Train dataset {dataset.__class__.__name__} (length: {len(dataset)})')

    if val_interval is not None:
        dataset_val_args = {k[4:]: v for k, v in config.items() if k.startswith('val_') and k != 'val_interval'}
        _, dataset_val_args, _ = filter_args(dataset_val_args, inspect.signature(dataset_cls).parameters)
        print('val args', {**dataset_args, **{'split': 'val', 'aug': 0}, **dataset_val_args})

        dataset_val = dataset_cls(**{**dataset_args, **{'split': 'val', 'aug': 0}, **dataset_val_args})

    # Optimizer setup
    opt_cls = get_attribute(config.optimizer)
    if config.optimize == 'torch.optim.SGD':
        opt_args = {'momentum': config.momentum if 'momentum' in config else 0}
    else:
        opt_args = {}
    opt = opt_cls(model.parameters(), lr=config.lr, **opt_args)

    # Learning rate scheduler
    if config.lr_scheduler == 'cosine':
        assert config.T_max is not None and config.eta_min is not None
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.T_max, config.eta_min)
    elif config.lr_scheduler == 'warmup_cosine':
        lr_scheduler = LambdaLR(opt, partial(cosine_warmup_lr, max_iter=(config.max_iterations), warmup=config.warmup))

    else:
        lr_scheduler = None

    batch_size, max_iterations = config.batch_size, config.max_iterations

    Dice_loss = diceloss.SoftDiceLoss()

    if config.amp:
        log.info('Using AMP')
        autocast_fn = autocast
        scaler = GradScaler()
    else:
        autocast_fn, scaler = nullcontext, None

    save_only_trainable = True
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=3, shuffle=False)

    tracker_config = config if not config.hyperparameter_optimization else None

    with (TrainingLogger(log_dir=config.name, model=model, config=tracker_config) as logger):

        i = 0
        train_losses = []
        val_losses = []
        while True:
            for data_x, data_y in data_loader:

                if config.mix:
                    assert config.mask.startswith('text_and')

                    with autocast_fn():
                        prompts = model.sample_prompts(data_x[1])
                        text_cond = model.compute_conditional(prompts)
                        visual_s_cond, _, _ = model.visual_forward(data_x[2].cuda())

                    max_txt = config.mix_text_max if config.mix_text_max is not None else 1
                    batch_size = text_cond.shape[0]
                    text_weights = torch.distributions.Uniform(config.mix_text_min, max_txt).sample((batch_size,))[:, None]
                    text_weights = text_weights.cuda()

                    if dataset.__class__.__name__ == 'PhraseCut':
                        visual_is_valid = data_x[3]
                        text_weights = torch.max(text_weights[:, 0], 1 - visual_is_valid.float().cuda()).unsqueeze(1)

                    cond = text_cond * text_weights + visual_s_cond * (1 - text_weights)

                else:
                    cond = data_x[1]
                    if isinstance(cond, torch.Tensor):
                        cond = cond.cuda()


                with autocast_fn():


                    pred = model(data_x[0].cuda(), cond, return_features=False)
                    loss = Dice_loss(pred, data_y[0].cuda())

                    if torch.isnan(loss) or torch.isinf(loss):
                        log.warning('Training stopped due to inf/nan loss.')
                        sys.exit(-1)

                    extra_loss = 0
                    loss += extra_loss

                opt.zero_grad()

                if scaler is None:
                    loss.backward()
                    opt.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                    if i % 25 == 0:
                        current_lr = [g['lr'] for g in opt.param_groups][0]
                        log.info(f'current lr: {current_lr:.5f} ({len(opt.param_groups)} parameter groups)')

                logger.iter(i=i, loss=loss)
                train_losses.append(float(loss))
                i += 1
                print(i)

                if i >= max_iterations:

                    if not isfile(join(logger.base_path, 'weights.pth')):
                        logger.save_weights(only_trainable=save_only_trainable)

                    save_loss_plot(train_losses, val_losses, val_interval, join(logger.base_path, 'loss_plot.png'))

                    sys.exit(0)

                if config.checkpoint_iterations is not None and i in config.checkpoint_iterations:
                    logger.save_weights(only_trainable=save_only_trainable, weight_file=f'weights_{i}.pth')

                if val_interval is not None and i % val_interval == val_interval - 1:

                    val_loss, val_scores, maximize = validate(model, dataset_val, config)

                    if len(val_scores) > 0:
                        score_str = f', scores: ' + ', '.join(f'{k}: {v}' for k, v in val_scores.items())

                        if maximize and val_scores[config.use_val_metric] > best_val_score:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_score = val_scores[config.use_val_metric]

                        elif not maximize and val_scores[config.use_val_metric] < best_val_score:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_score = val_scores[config.use_val_metric]

                    else:
                        score_str = ''
                        if val_loss < best_val_loss:
                            logger.save_weights(only_trainable=save_only_trainable)
                            best_val_loss = val_loss

                    log.info(f'Validation loss: {val_loss}' + score_str)
                    logger.iter(i=i, val_loss=val_loss, extra_loss=float(extra_loss), **val_scores)
                    val_losses.append(val_loss)
                    model.train()

                    save_loss_plot(train_losses, val_losses, val_interval, join(logger.base_path, 'loss_plot.png'))

            save_loss_plot(train_losses, val_losses, val_interval, join(logger.base_path, 'loss_plot.png'))
            print('epoch complete')

if __name__ == '__main__':
    main()
    print("ok")
