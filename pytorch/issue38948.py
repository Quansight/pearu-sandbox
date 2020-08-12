import torch as T
import sys

class LOBPCG2(T.autograd.Function):

    @staticmethod
    def forward(ctx, A):
        #n = int(len(S) ** 0.5)
        #A = S.reshape(n, n)
        print('A=', A)
        n = len(A)
        k = n
        if 1:
            e, v = T.symeig(-A, eigenvectors=True)
            e = -e[:k]
            v = v[:, :k]
        else:
            e, v = T.lobpcg(A, k=k)
        r = T.cat((T.flatten(v), e), 0)
        ctx.save_for_backward(e, v, A)
        print('r=', r)
        return e, v

    @staticmethod
    def backward(ctx, de, dv):
        """
        solve `dA v + A dv = dv diag(e) + v diag(de)` for `dA`


        A.grad = U (D.grad + (U^T U.grad o F)) U^T
        """
        e, v, S = ctx.saved_tensors
        n, k = v.shape
        A = S.reshape(n, n)

        print('e=', e)
        vt = v.transpose(-2, -1)
        print('vt=', vt)
        print('de=', de)
        print('dv=', dv)

        if dv is None:
            A_bar = T.mm(v, T.mm(T.diag(de), vt))
        else:
            vtdv = T.mm(vt, dv)
            print('vtdv=', vtdv)
            F = T.ones_like(vtdv) * e
            F = (F - F.transpose(-2, -1)) ** -1
            F.diagonal().fill_(0)

            print('F=',F)

            A_bar = T.mm(v, T.mm(T.diag(de) + F * vtdv, vt))


        for i in range(k):
            break
            for j in range(k):
                if i < j:
                    A_bar[i,j] *= 2
                elif i>j:
                    A_bar[i,j] *= 0
        A_bar = (A_bar + A_bar.transpose(-2, -1))/2
        print('A_bar=', A_bar)        
        return A_bar
        S_bar = A_bar.flatten()
        return S_bar

import torch as T
T.random.manual_seed(123)
A = T.randn(2, 2).double()
S = A.matmul(A.t())
S = S.double().detach().requires_grad_(True)
def mysymeig(A):
    return T.symeig(A, eigenvectors=True)
#T.autograd.gradcheck(mysymeig, S)
T.autograd.gradcheck(LOBPCG2.apply, S)
