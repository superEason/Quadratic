import numpy as np
from quadratic.layers import softmax_loss


class QuadraticNetwork(object):
    def __init__(self, input_size, num_classes, weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg
        self.D = input_size
        self.C = num_classes
        self.reg = reg
        self.params['Wr'] = weight_scale * np.random.randn(self.D, self.C)
        self.params['br'] = np.zeros(self.C)
        self.params['Wg'] = weight_scale * np.random.randn(self.D, self.C)
        self.params['bg'] = np.zeros(self.C)
        self.params['Wb'] = weight_scale * np.random.randn(self.D, self.C)
        self.params['b'] = np.zeros(self.C)

    def loss(self, X, y=None):
        Wr, br = self.params['Wr'], self.params['br']
        Wg, bg = self.params['Wg'], self.params['bg']
        Wb, b = self.params['Wb'], self.params['b']
        N, D = X.shape      
        h = (X.dot(Wr)+br)*(X.dot(Wg)+bg)+(X*X).dot(Wb)+b
        scores = np.maximum(0, h)
        if y is None:
            return scores
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (np.sum(Wr**2)+np.sum(Wg**2)+np.sum(Wb**2))
        loss = data_loss + reg_loss

        grads = {}
        dh = dscores * (h >= 0)
        dWr = 2 * np.dot(X.T, (X.dot(Wg)+bg)*dh)
        dWg = 2 * np.dot(X.T, (X.dot(Wr)+br)*dh)
        dbr = 2 * np.sum(dh * (X.dot(Wg)+bg), axis=0)
        dbg = 2 * np.sum(dh * (X.dot(Wr)+br), axis=0)
        dWb = np.dot(X.T*X.T, dh)
        db = np.sum(dh, axis=0)

        grads.update({'Wr': dWr,
                      'br': dbr,
                      'Wg': dWg,
                      'bg': dbg,
                      'Wb': dWb,
                      'b': db})

        return loss, grads
