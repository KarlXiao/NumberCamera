import torch.nn.functional as F


def criterion(length_logits, digits_logits, length_labels, digits_labels):
    length = F.cross_entropy(length_logits, length_labels, reduction='mean')
    digit0 = F.cross_entropy(digits_logits[:, 0], digits_labels[:, 0], reduction='mean')
    digit1 = F.cross_entropy(digits_logits[:, 1], digits_labels[:, 1], reduction='mean')
    digit2 = F.cross_entropy(digits_logits[:, 2], digits_labels[:, 2], reduction='mean')
    digit3 = F.cross_entropy(digits_logits[:, 3], digits_labels[:, 3], reduction='mean')
    digit4 = F.cross_entropy(digits_logits[:, 4], digits_labels[:, 4], reduction='mean')
    loss = length + digit0 + digit1 + digit2 + digit3 + digit4
    return loss
