import numpy as np


def sgd(x: np.ndarray,
        y: np.ndarray,
        lr: float,
        loss_fn, grad_loss_fn,
        smart_init=False,
        w: np.ndarray = None,
        lam: float = 0.9,
        eps = 1e-6,
        batch_size=1,
        max_iter=10000) -> (np.ndarray, float):

    n_features = x.shape[1]
    n_objects = x.shape[0]

    if w is None:
        if not smart_init:
            w = np.zeros(n_features)
            for i in range(n_features):
                w[i] = np.random.normal(0, 1 / (2 * x.shape[0]))

    w = np.expand_dims(w, -1)


    delta_q = float("inf")
    delta_w = np.array([float("inf")] * n_features)
    loss = loss_fn(w, x, y)
    q = np.mean(loss)

    iter = 0
    while iter < max_iter and (abs(delta_q) > eps or abs(np.sum(delta_w)) > (eps * n_features)):

        idx_obj = np.random.choice(range(n_objects), size=batch_size)
        x_obj = x[idx_obj,:]
        y_obj = y[idx_obj]

        loss = np.mean(loss_fn(w, x_obj, y_obj))
        delta_w = lr * grad_loss_fn(w, x_obj, y_obj)
        w = w - delta_w

        new_q = lam * loss + (1 - lam) * q
        delta_q = new_q - q
        q = new_q

        iter += 1

    return w, q













