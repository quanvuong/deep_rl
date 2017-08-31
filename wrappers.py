import torch
from torch.autograd import Variable

FloatTensor = lambda x: torch.FloatTensor(x)
FloatTensorFromNumpy = lambda x: torch.from_numpy(x).float()

ZeroTensor = lambda *s: torch.zeros(*s)
ZeroTensorVar = lambda *tensor: Variable(ZeroTensor(*tensor))

ByteTensor = lambda x: torch.ByteTensor(x)
ByteTensorFromNumpy = lambda x: torch.from_numpy(x).byte()

IntTensor = lambda x: torch.from_numpy(x).int()
IntTensorVar = lambda x: Variable(IntTensor(x))


def FloatTensorVar(x, args, requires_grad=False):
    if args.use_cuda:
        return Variable(FloatTensor(x).cuda(), requires_grad)
    return Variable(FloatTensor(x), requires_grad)


def FloatTensorFromNumpyVar(x, args):
    if args.use_cuda:
        return Variable(FloatTensorFromNumpy(x)).cuda()
    return Variable(FloatTensorFromNumpy(x))


def ByteTensorVar(x):
    return Variable(ByteTensor(x))


def ByteTensorFromNumpyVar(x):
    return Variable(ByteTensorFromNumpy(x))
