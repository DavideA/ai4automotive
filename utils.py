import matplotlib.cm as cm
import numpy as np
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
eps = np.finfo(np.float32).eps


def set_random_seed(worker_id):
    np.random.seed(worker_id)


def to_image(p_res, y_res, x_ff):
    cmap = cm.get_cmap('jet')
    p_res, y_res, x_ff = (t.to('cpu').numpy().transpose(0, 2, 3, 1)
                          for t in (p_res, y_res, x_ff))
    p_res /= np.apply_over_axes(np.max, p_res, [1, 2, 3]) + eps
    y_res /= np.apply_over_axes(np.max, y_res, [1, 2, 3]) + eps
    p_res = np.concatenate(list(p_res), axis=1)
    y_res = np.concatenate(list(y_res), axis=1)
    p_res_rgb = cmap(np.squeeze(p_res))[..., :3]
    y_res_rgb = cmap(np.squeeze(y_res))[..., :3]
    x_ff = np.concatenate(list(x_ff), axis=1)
    image = np.concatenate(
        (
            x_ff,
            0.5 * x_ff + 0.5 * p_res_rgb,
            0.5 * x_ff + 0.5 * y_res_rgb
        ), axis=0
    )
    return image
