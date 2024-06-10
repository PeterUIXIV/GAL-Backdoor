import numpy as np
from config import cfg

from marks.image import DCT, IDCT, RGB2YUV, YUV2RGB


def poison(x_train, y_train):
    target_label = cfg["target_class"]
    num_images = int(cfg["poison_percent"] * y_train.shape[0])

    index = np.where(y_train != target_label)
    index = index[0]
    index = index[:num_images]
    x_train[index] = poison_frequency(x_train[index])
    y_train[index] = target_label
    return x_train


def poison_frequency(x_train):
    window_size = cfg['data_shape'][1]
    if x_train.shape[0] == 0:
        return x_train

    x_train *= 255.
    if cfg["YUV"]:
        x_train = RGB2YUV(x_train)

    # transfer to frequency domain
    x_train = DCT(x_train, window_size)  # (idx, ch, w, h)

    # plug trigger frequency
    for i in range(x_train.shape[0]):
        for ch in cfg["channel_list"]:
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    for pos in cfg["pos_list"]:
                        x_train[i][ch][w + pos[0]][h + pos[1]] += cfg["magnitude"]


    x_train = IDCT(x_train, window_size)  # (idx, w, h, ch)

    if cfg["YUV"]:
        x_train = YUV2RGB(x_train)

    x_train /= 255.
    x_train = np.clip(x_train, 0, 1)
    return x_train