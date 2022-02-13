import numpy as np


def sgd(x: np.ndarray,
        y: np.ndarray,
        loss_fn, grad_loss_fn,
        lr = 1e-4,
        smart_init=False,
        w: np.ndarray = None,
        lam: float = 0.9,
        eps=1e-6,
        batch_size=1,
        max_iter=10000,
        logging=False) -> (np.ndarray, float):

    n_features = x.shape[1]
    n_objects = x.shape[0]

    if w is None:
        if not smart_init:
            w = np.zeros(n_features)
            for i in range(n_features):
                w[i] = np.random.normal(0, 1 / (2 * x.shape[0]))

        w = np.expand_dims(w, -1)

    # loss = loss_fn(w, x, y)
    loss = float('inf')

    history = []
    iter = 0

    if logging:
        print(f"Iter = {iter}, Loss = {loss}")

    while abs(loss) > eps and iter < max_iter:# or abs(np.sum(delta_w)) > (eps * n_features)):

        idx_obj = np.random.choice(range(n_objects), size=batch_size)
        x_obj = x[idx_obj,:]
        y_obj = y[idx_obj]

        loss = np.sum(loss_fn(w, x_obj, y_obj)) / x_obj.shape[0]

        grad = grad_loss_fn(w, x_obj, y_obj)

        delta_w = lr * grad
        w = w - delta_w

        loss = np.mean(loss_fn(w, x_obj, y_obj))

        #print(f"iter {iter}\ngrad = {grad}\ndelta w = {delta_w}\ndelta q = {delta_q}\n\n")
        iter += 1
        history.append(loss)

        if logging:
            print(f"Iter = {iter}, Loss = {loss}, delta_w = {delta_w}")

    return w, loss, history


class SGDOptimizer:

    def __init__(self,
                 lr: float =1e-4,
                 smart_init=False,
                 lam: float = 0.9,
                 eps = 1e-6,
                 max_iter = 10000,
                 logging=False,
                 batch_size=1):

        self.lr = lr
        self.smart_init = smart_init
        self.lam = lam
        self.eps = eps
        self.max_iter = max_iter
        self.logging = logging
        self.bs = batch_size

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            loss_fn,
            grad_loss_fn,
            w: np.ndarray = None,
            get_hist=False) -> (np.ndarray, float):
        # weight and quality

        res = sgd(x, y, loss_fn, grad_loss_fn,
                   self.lr, self.smart_init, w, self.lam, self.eps, self.bs, self.max_iter, self.logging)

        if get_hist:
            return res

        else:
            return res[0], res[1]