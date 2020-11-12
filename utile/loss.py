import torch
import torch.nn as nn


class SegmentationLosses(object):
    def __init__(self,
                 weight=None,
                 batch_average=True,
                 ignore_index=255,
                 cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """"""
        if mode == 'ce':
            return self.cross_entropy_loss

    def cross_entropy_loss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        # target should be long type because the crossentropy function
        # In the formula, target is used to index the output logit for
        # the current target class(note the indexing in x[class])
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def focal_loss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction='mean')

        if self.cuda:
            criterion = criterion.cuda()
        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt)**gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


if __name__ == "__main__":
    loss =SegmentationLosses(cuda=True)
    a = torch.rand(1,3,7,7).cuda()
    b =  torch.rand(1,7,7).cuda()
    print(loss.cross_entropy_loss(a,b).item())
    print(loss.focal_loss(a, b, gamma=0, alpha=None).item())
    print(loss.focal_loss(a, b, gamma=2, alpha=0.5).item())