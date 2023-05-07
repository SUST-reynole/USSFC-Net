import logging
import torch
from PIL import Image
import numpy as np


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def to8bits(img):
    result = np.ones([img.shape[0], img.shape[1]], dtype='int')
    result[img == 0] = 0
    result[img == 1] = 255
    return result


def save_pre_result(pre, flag, num, save_path):
    pre[pre >= 0.5] = 255
    pre[pre < 0.5] = 0
    outputs = torch.squeeze(pre).cpu().detach().numpy()
    outputs = Image.fromarray(np.uint8(outputs))
    outputs.save(save_path + '/%s_%d.png' % (flag, num))