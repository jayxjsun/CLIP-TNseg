import torch
import inspect
import json
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import metrics
import numpy as np
from os.path import join, isfile, realpath
from torch.utils.data import DataLoader

from general_utils import load_model, log, score_config_from_cli_args, AttributeDict, get_attribute, filter_args

DATASET_CACHE = dict()


def load_model(checkpoint_id, weights_file=None, strict=True, model_args='from_config', with_config=False,
               ignore_weights=False):
    config = json.load(open(join('logs', checkpoint_id, 'config.json')))

    if model_args != 'from_config' and type(model_args) != dict:
        raise ValueError('model_args must either be "from_config" or a dictionary of values')

    model_cls = get_attribute(config['model'])

    # load model
    if model_args == 'from_config':
        _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)

    model = model_cls(**model_args)

    if weights_file is None:
        weights_file = realpath(join('logs', checkpoint_id, 'weights.pth'))
    else:
        weights_file = realpath(join('logs', checkpoint_id, weights_file))

    if isfile(weights_file) and not ignore_weights:
        weights = torch.load(weights_file)
        for _, w in weights.items():
            assert not torch.any(torch.isnan(w)), 'weights contain NaNs'
        model.load_state_dict(weights, strict=strict)
    else:
        if not ignore_weights:
            raise FileNotFoundError(f'model checkpoint {weights_file} was not found')

    if with_config:
        return model, config

    return model

def main():
    config, train_checkpoint_id = score_config_from_cli_args()

    config = AttributeDict(config)

    print(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # use training dataset and loss
    train_config = AttributeDict(json.load(open(f'logs/{train_checkpoint_id}/config.json')))

    cp_str = f'_{config.iteration_cp}' if config.iteration_cp is not None else ''

    model_cls = get_attribute(train_config['model'])
    _, model_args, _ = filter_args(train_config, inspect.signature(model_cls).parameters)
    model_args = {**model_args, **{k: config[k] for k in ['process_cond', 'fix_shift'] if k in config}}

    model = load_model(train_checkpoint_id, strict=False, model_args=model_args, weights_file=f'weights{cp_str}.pth')

    model.eval()
    model.cuda()

    metric_args = dict()
    metric_args['sigmoid'] = config.sigmoid


    if config.test_dataset == 'phrasecut':
        from datasets.thyroid import PhraseCut
        dataset = PhraseCut('test', image_size=352, mask=config.mask, with_visual=config.with_visual,
                            only_visual=config.only_visual, aug_crop=False, aug_color=False)
        loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, drop_last=False)

        savepath = f'logs/{train_checkpoint_id}/preout/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            os.makedirs(savepath + "mask/")

        mean_acc = 0
        mean_dice = 0
        mean_iou = 0
        background_iou = 0
        foreground_iou = 0
        pbar = tqdm(enumerate(loader), total=len(dataset) // loader.batch_size)

        for iteration, (datax, datay) in pbar:
            with torch.no_grad():

                datax[0] = datax[0].to(device)
                datax[1] = tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in datax[1])
                datay[0] = datay[0].to(device)
                img_np = datax[0]

                pred = model(datax[0], datax[1], return_features=False)
                preds = torch.sigmoid(pred)
                accs, miou, mdice, _, iou_calss, pred_save = metrics.calculate_metrics(preds.cpu(), datay[0][0].long().cpu())
                datay_np = datay[0][0].cpu().numpy()[0]
                img_np_transposed = img_np.cpu().numpy()[0].transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_denormalized = img_np_transposed * std + mean
                img_denormalized = np.clip(img_denormalized, 0, 1)

            imgs_list = [img_denormalized, pred_save, datay_np]
            titles = ["img", "pre", "label"]
            title = "label_%s.png: acc: %.5f | iou: %.5f | dice: %.5f" % (iteration, accs, miou, mdice)
            for i in range(3):
                ax = plt.subplot(1, 3, i + 1)
                plt.xticks([]), plt.yticks([])
                plt.imshow(imgs_list[i], 'gray' if i > 0 else None)
                ax.set_title(titles[i])
            plt.suptitle(title)
            # plt.savefig(savepath + "label_%s.png" % iteration)
            plt.close()
            # plt.show()
            # cv2.imwrite(savepath + "mask/"+"pre_%s.png" % iteration, pred_save * 255)
            # cv2.imwrite(savepath + "mask/" + "label_%s.png" % iteration, datay_np * 255)

            mean_acc += accs
            mean_dice += mdice
            mean_iou += miou
            background_iou += iou_calss[0]
            foreground_iou += iou_calss[1]

        mean_acc /= iteration + 1
        mean_dice /= iteration + 1
        mean_iou /= iteration + 1
        background_iou /= iteration + 1
        foreground_iou /= iteration + 1
        # print(background_iou)
        # print(foreground_iou)
        # print(mean_iou)
        # print(mean_acc)
        # print(mean_dice)

        message = """The prediction results with %s: mean_acc: %.5f | mean_iou: %.5f | mean_dice: %.5f | background_iou: %.5f | foreground_iou: %.5f""" \
                  % (train_checkpoint_id, mean_acc, mean_iou, mean_dice, background_iou, foreground_iou)
        with open(savepath + "results.txt", 'w') as f:
            f.write(message)



if __name__ == '__main__':
    main()