import numpy as np


def sgd(x: np.ndarray,
        y: np.ndarray,
        loss_fn, grad_loss_fn,
        lr = 1e-4,
        smart_init=False,
        w: np.ndarray = None,
        lam: float = 0.9,
        eps=1e-6,
        batch_size=1, # менять bs нельзя, данный метод не работает для нескольких элементов в батче
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

    grad_coef = np.asarray([grad_loss_fn(w, np.expand_dims(x[i], axis=0), y[i]) for i in range(n_objects)])
    grad_sum = np.sum(grad_coef, axis=0)


    delta_q = float("inf")
    delta_w = np.array([float("inf")] * n_features)
    loss = loss_fn(w, x, y)
    q = np.sum(loss) / x.shape[0]

    history = []
    iter = 0

    if logging:
        print(f"Iter = {iter}, Loss = {q}")

    while iter < max_iter and abs(delta_q) > eps:# or abs(np.sum(delta_w)) > (eps * n_features)):

        idx_obj = np.random.choice(range(n_objects), size=batch_size)
        x_obj = x[idx_obj,:]
        y_obj = y[idx_obj]

        loss = np.sum(loss_fn(w, x_obj, y_obj)) / x_obj.shape[0]

        grad = grad_loss_fn(w, x_obj, y_obj)
        grad_sum -= grad_coef[idx_obj[0]]
        grad_coef[idx_obj[0]] = grad
        grad_sum += grad_coef[idx_obj[0]]

        delta_w = lr * grad_sum / n_objects
        w = w - delta_w

        new_q = lam * loss + (1 - lam) * q
        delta_q = new_q - q
        q = new_q

        #print(f"iter {iter}\ngrad = {grad}\ndelta w = {delta_w}\ndelta q = {delta_q}\n\n")
        iter += 1
        history.append(q)

        if logging:
            print(f"Iter = {iter}, Loss = {q}")

    return w, q, history


class SGDOptimizer:

    def __init__(self,
                 lr: float =1e-4,
                 smart_init=False,
                 w: np.ndarray = None,
                 lam: float = 0.9,
                 eps = 1e-6,
                 max_iter = 10000,
                 logging=False):

        self.lr = lr
        self.smart_init = smart_init
        self.w = w
        self.lam = lam
        self.eps = eps
        self.max_iter = max_iter
        self.logging = logging

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            loss_fn,
            grad_loss_fn,
            get_hist=False) -> (np.ndarray, float):
        # weight and quality

        res = sgd(x, y, loss_fn, grad_loss_fn,
                   self.lr, self.smart_init, self.w, self.lam, self.eps, 1, self.max_iter, self.logging)

        if get_hist:
            return res

        else:
            return res[0], res[1]
