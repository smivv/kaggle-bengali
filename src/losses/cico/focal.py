from torch.nn import functional as F

from catalyst.contrib.nn import FocalLossBinary


class FocalLossMultiClassFixed(FocalLossBinary):
    """
    Compute focal loss for multi-class problem.
    Ignores targets having -1 label
    """

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; num_classes; ...]
            targets: [bs; ...] or [bs; num_classes; ...]
        """
        num_classes = logits.size(1)
        loss = 0

        if len(targets.size()) == 2:
            targets = targets.argmax(dim=1)

        targets = targets.view(-1)
        logits = logits.view(-1, num_classes)

        # Filter anchors with -1 label from loss computation
        if self.ignore is not None:
            not_ignored = targets != self.ignore

        for cls in range(num_classes):
            cls_label_target = (targets == (cls + 0)).long()
            cls_label_input = logits[..., cls]

            if self.ignore is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += self.loss_fn(cls_label_input, cls_label_target)

        return loss

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

