import tensorflow as tf

def binary_focal_loss(gamma=2):
    from focal_loss import BinaryFocalLoss
    return BinaryFocalLoss(gamma=gamma)
