import torch
import torch.nn as nn

from catalyst.contrib.nn.criterion.functional import euclidean_distance,\
    cosine_distance


class CenterLoss(nn.Module):
    """
    Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        embedding_size (int): feature dimension.
    """

    def __init__(self, num_classes=None, embedding_size=None, metric="cosine"):
        super(CenterLoss, self).__init__()

        assert metric in ["euclidean", "cosine"]

        self.metric_fn = euclidean_distance \
            if metric == "euclidean" else cosine_distance

        self.num_classes = num_classes
        self.embedding_size = embedding_size
        if num_classes is None or embedding_size is None:
            self.centers = None
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.embedding_size))

    def forward(self, features, targets):
        """
        Args:
            features: feature matrix with shape (batch_size, feat_dim).
            targets: ground truth labels with shape (batch_size).
        """
        batch_size = features.size(0)

        if self.centers is None:
            assert len(targets.size()) == 2, "Targets should be one hot!"
            self.num_classes = targets.size()[1]
            self.embedding_size = features.size()[1]
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.embedding_size))

        distmat = self.metric_fn(features, self.centers.to(features.device))

        classes = torch.arange(self.num_classes).long().to(targets.device)

        if len(targets.size()) == 2:
            targets = targets.argmax(dim=1)

        targets = targets.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = targets.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss