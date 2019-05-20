from os.path import join

import numpy as np
import skimage.io as io
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

from utils import set_random_seed


class Preprocess:

    def __init__(self, raw_shape, small_shape, pre_crop_shape):
        self.raw_shape = raw_shape
        self.small_shape = small_shape
        self.pre_crop_shape = pre_crop_shape

    def __call__(self, sample):
        x, y = sample
        t, c, h, w = self.raw_shape
        t, c, ho, wo = self.small_shape
        t, c, hpc, wpc = self.pre_crop_shape

        x = np.stack([resize(x, (h, w)) for x in x], axis=0)
        y = resize(y[..., 0], (h, w))[..., np.newaxis]

        x_ff = x[-1]
        x_ff = np.transpose(x_ff, (2, 0, 1))
        x_ff = torch.FloatTensor(x_ff)

        x_res = np.stack([resize(x, (ho, wo)) for x in x], axis=0)
        x_res = np.transpose(x_res, (3, 0, 1, 2))
        x_res = torch.FloatTensor(x_res)

        x_crp = np.stack([resize(x, (hpc, wpc)) for x in x], axis=0)
        hc = np.random.randint(0, hpc - ho)
        wc = np.random.randint(0, wpc - wo)
        x_crp = x_crp[:, hc:hc+ho, wc:wc+wo, :]
        x_crp = np.transpose(x_crp, (3, 0, 1, 2))
        x_crp = torch.FloatTensor(x_crp)
        y_crp = resize(y[..., 0], (hpc, wpc))[..., np.newaxis]
        y_crp = y_crp[hc:hc+ho, wc:wc+wo]
        y_crp = np.transpose(y_crp, (2, 0, 1))
        y_crp = torch.FloatTensor(y_crp)

        y = np.transpose(y, (2, 0, 1))
        y = torch.FloatTensor(y)

        return x_res, x_crp, x_ff, y, y_crp


class DREYEVE(Dataset):

    path = '/nas/majinbu/DREYEVE/'
    data_path = '/nas/majinbu/DREYEVE/DATA'
    train_seqs = list(range(1, 37 + 1))
    test_seqs = list(range(38, 74 + 1))
    n_sequences = 74
    sequence_length = 7500

    train_samples_per_epoch = 8192

    TRAIN_MODE = 0
    TEST_MODE = 1

    def __init__(self):
        super(DREYEVE, self).__init__()

        self.mode = None
        self.test_seq = None
        self.preprocess = Preprocess(
            self.shape,
            self.small_shape,
            self.pre_crop_shape
        )

        self.train_record_path = join(self.path, 'ade_binary_records',
                                      'train.bin')
        self.train_record_path_fix = join(self.path, 'ade_binary_records',
                                      'train_fix.bin')
        self.cur_record_rgb = None
        self.cur_record_fix = None

    def train(self):
        self.mode = DREYEVE.TRAIN_MODE
        self.test_seq = None
        self.cur_record_rgb = open(self.train_record_path, mode='rb')
        self.cur_record_fix = open(self.train_record_path_fix, mode='rb')

    def test(self, test_seq):
        self.mode = DREYEVE.TEST_MODE
        self.test_seq = test_seq

        for file in [self.cur_record_rgb, self.cur_record_fix]:
            try:
                file.close()
            except:
                continue

    def __len__(self):
        self._check_mode()
        if self.mode == DREYEVE.TRAIN_MODE:
            return DREYEVE.train_samples_per_epoch
        if self.mode == DREYEVE.TEST_MODE:
            c, t, h, w = self.shape
            return DREYEVE.sequence_length - t + 1

    @property
    def shape(self):
        return 3, 16, 448, 448

    @property
    def small_shape(self):
        return 3, 16, 112, 112

    @property
    def pre_crop_shape(self):
        return 3, 16, 256, 256

    def __getitem__(self, idx):
        self._check_mode()

        c, t, h, w = self.shape
        x = np.zeros(shape=(t, h, w, c))
        y = np.zeros(shape=(h, w, 1))
        if self.mode == DREYEVE.TRAIN_MODE:
            _, _ ,rh, rw = 3, 16, 160, 256
            while True:
                try:
                    seq = np.random.choice(DREYEVE.train_seqs)
                    start = np.random.randint(0, DREYEVE.sequence_length - t + 1)
                    offset = (DREYEVE.sequence_length * seq + start) * (rh * rw * c)
                    self.cur_record_rgb.seek(offset)
                    x = np.fromfile(self.cur_record_rgb, dtype=np.uint8,
                                    count=(t * rh * rw * c))
                    x = np.reshape(x, newshape=(t, rh, rw, c))
                    offset = (DREYEVE.sequence_length * seq + start + t) * (rh * rw)
                    self.cur_record_fix.seek(offset)
                    y = np.fromfile(self.cur_record_fix, dtype=np.uint8, count=(rh * rw))
                    y = np.reshape(y, newshape=(rh, rw, 1))
                    break
                except ValueError:
                    continue

        if self.mode == DREYEVE.TEST_MODE:
            seq = self.test_seq
            start = idx
            # Load data
            for i in range(0, t):
                index = start + i
                img = io.imread(join(DREYEVE.data_path, f'{seq:02d}',
                                     'frames', f'{index:06d}.jpg'))
                img = resize(img, (h, w), preserve_range=False)
                x[i] = img
            gt = io.imread(join(DREYEVE.data_path, f'{seq:02d}',
                                'saliency_fix', f'{start + t - 1:06d}.png'),
                           as_gray=True)
            gt = resize(gt, (h, w), preserve_range=False)
            y[..., 0] = gt

        return self.preprocess((x, y))

    def _check_mode(self):
        assert self.mode in [DREYEVE.TRAIN_MODE, DREYEVE.TEST_MODE]


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    plt.ion()
    fig, (axes) = plt.subplots(1, 5)

    ds = DREYEVE()
    ds.train()
    dl = DataLoader(ds, num_workers=8, worker_init_fn=set_random_seed)
    for x_res, x_crp, x_ff, y, y_crp in dl:
        x_res, x_crp, x_ff, y, y_crp = [
            tensor.squeeze(0).numpy()
            for tensor in (x_res, x_crp, x_ff, y, y_crp)]

        c, t, h, w = ds.shape
        x_res = np.transpose(x_res, (1, 2, 3, 0))
        x_crp = np.transpose(x_crp, (1, 2, 3, 0))
        x_ff = np.transpose(x_ff, (1, 2, 0))
        y = np.transpose(y, (1, 2, 0))
        y_crp = np.transpose(y_crp, (1, 2, 0))
        for i in range(0, t):
            for ax, im in zip(axes, [x_res[i], x_crp[i], x_ff, y, y_crp]):
                ax.cla()
                ax.imshow(np.squeeze(im))
            plt.pause(0.04)
