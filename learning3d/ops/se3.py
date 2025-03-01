""" 3-d rigid body transfomation group and corresponding Lie algebra. """
import torch
from .sinc import sinc1, sinc2, sinc3
from . import so3

def twist_prod(x, y):
    x_ = x.view(-1, 6)
    y_ = y.view(-1, 6)

    xw, xv = x_[:, 0:3], x_[:, 3:6]
    yw, yv = y_[:, 0:3], y_[:, 3:6]

    zw = so3.cross_prod(xw, yw)
    zv = so3.cross_prod(xw, yv) + so3.cross_prod(xv, yw)

    z = torch.cat((zw, zv), dim=1)

    return z.view_as(x)

def liebracket(x, y):
    return twist_prod(x, y)


def mat(x):
    # size: [*, 6] -> [*, 4, 4]
    x_ = x.view(-1, 6)
    w1, w2, w3 = x_[:, 0], x_[:, 1], x_[:, 2]
    v1, v2, v3 = x_[:, 3], x_[:, 4], x_[:, 5]
    O = torch.zeros_like(w1)

    X = torch.stack((
        torch.stack((  O, -w3,  w2, v1), dim=1),
        torch.stack(( w3,   O, -w1, v2), dim=1),
        torch.stack((-w2,  w1,   O, v3), dim=1),
        torch.stack((  O,   O,   O,  O), dim=1)), dim=1)
    return X.view(*(x.size()[0:-1]), 4, 4)

def vec(X):
    X_ = X.view(-1, 4, 4)
    w1, w2, w3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
    v1, v2, v3 = X_[:, 0, 3], X_[:, 1, 3], X_[:, 2, 3]
    x = torch.stack((w1, w2, w3, v1, v2, v3), dim=1)
    return x.view(*X.size()[0:-2], 6)

def genvec():
    return torch.eye(6)

def genmat():
    return mat(genvec())

def exp(x):
    x_ = x.view(-1, 6)
    w, v = x_[:, 0:3], x_[:, 3:6]
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = so3.mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)

    # Rodrigues' rotation formula.
    #R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
    #  = eye(3) + sinc1(t)*W + sinc2(t)*S
    R = I + sinc1(t)*W + sinc2(t)*S

    #V = sinc1(t)*eye(3) + sinc2(t)*W + sinc3(t)*(w*w')
    #  = eye(3) + sinc2(t)*W + sinc3(t)*S
    V = I + sinc2(t)*W + sinc3(t)*S

    p = V.bmm(v.contiguous().view(-1, 3, 1))

    z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(x_.size(0), 1, 1).to(x)
    Rp = torch.cat((R, p), dim=2)
    g = torch.cat((Rp, z), dim=1)

    return g.view(*(x.size()[0:-1]), 4, 4)

def inverse(g):
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3]
    p = g_[:, 0:3, 3]
    Q = R.transpose(1, 2)
    q = -Q.matmul(p.unsqueeze(-1))

    z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(g_.size(0), 1, 1).to(g)
    Qq = torch.cat((Q, q), dim=2)
    ig = torch.cat((Qq, z), dim=1)

    return ig.view(*(g.size()[0:-2]), 4, 4)


def log(g):
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3]
    p = g_[:, 0:3, 3]

    w = so3.log(R)
    H = so3.inv_vecs_Xg_ig(w)
    v = H.bmm(p.contiguous().view(-1, 3, 1)).view(-1, 3)

    x = torch.cat((w, v), dim=1)
    return x.view(*(g.size()[0:-2]), 6)

def transform(g, a):
    # g : SE(3),  * x 4 x 4
    # a : R^3,    * x 3[x N]
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3].contiguous().view(*(g.size()[0:-2]), 3, 3)
    p = g_[:, 0:3, 3].contiguous().view(*(g.size()[0:-2]), 3)
    if len(g.size()) == len(a.size()):
        b = R.matmul(a) + p.unsqueeze(-1)
    else:
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p
    return b

def group_prod(g, h):
    # g, h : SE(3)
    g1 = g.matmul(h)
    return g1


class ExpMap(torch.autograd.Function):
    """ Exp: se(3) -> SE(3)
    """
    @staticmethod
    def forward(ctx, x):
        """ Exp: R^6 -> M(4),
            size: [B, 6] -> [B, 4, 4],
              or  [B, 1, 6] -> [B, 1, 4, 4]
        """
        ctx.save_for_backward(x)
        g = exp(x)
        return g

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        g = exp(x)
        gen_k = genmat().to(x)

        # Let z = f(g) = f(exp(x))
        # dz = df/dgij * dgij/dxk * dxk
        #    = df/dgij * (d/dxk)[exp(x)]_ij * dxk
        #    = df/dgij * [gen_k*g]_ij * dxk

        dg = gen_k.matmul(g.view(-1, 1, 4, 4))
        # (k, i, j)
        dg = dg.to(grad_output)

        go = grad_output.contiguous().view(-1, 1, 4, 4)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)

        return grad_input

Exp = ExpMap.apply


#EOF
