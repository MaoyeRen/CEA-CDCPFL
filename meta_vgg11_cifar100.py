import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
import  numpy as np

from    learner import Learner


class Meta_vgg11_cifar100(nn.Module):

    def __init__(self, args):

        super(Meta_vgg11_cifar100, self).__init__()
        config = [

            ('conv2d', [64, 3, 3, 3, 1, 1]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),

            ('conv2d', [128, 64, 3, 3, 1, 1]),
            ('relu', [True]),

            ('max_pool2d', [2, 2, 0]),

            ('conv2d', [256, 128, 3, 3, 1, 1]),
            ('relu', [True]),
            ('conv2d', [256, 256, 3, 3, 1, 1]),
            ('relu', [True]),
            ('max_pool2d', [2,2,0]),

            ('conv2d', [512, 256, 3, 3, 1, 1]),
            ('relu', [True]),
            ('conv2d', [512, 512, 3, 3, 1, 1]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),

            ('conv2d', [512, 512, 3, 3, 1, 1]),
            ('relu', [True]),
            ('conv2d', [512, 512, 3, 3, 1, 1]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),
            ('adaptive_avg_pool2d', [1,1]),
            ('flatten', []),

            ('linear', [ 1024, 512 * 1 * 1]),
            ('relu', [True]),
            ('linear', [1024, 1024]),
            ('relu', [True]),
            ('linear', [args.num_classes, 1024])
        ]

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.task_num = 1
        self.update_step = 1
        self.update_step_test = 1
        self.net = Learner(config)
        self.meta_optim = optim.SGD(self.net.parameters(), lr=self.meta_lr,
                                        momentum=0.9)

    def forward(self, x_spt, y_spt):

        task_num = 1


        for i in range(task_num):

            logits = self.net(x_spt, vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt.long())
            grad = torch.autograd.grad(loss, self.net.parameters())
            with torch.no_grad():
                fast_weights = list( map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        accs=0

        return accs, loss,  fast_weights, grad

    def finetunning(self, x_qry, y_qry):

        task_num = 1
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step + 1)]

        with torch.no_grad():

            logits_q = self.net(x_qry, self.net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

            corrects[1]=0

        accs = np.array(corrects) / (querysz * task_num)

        return accs