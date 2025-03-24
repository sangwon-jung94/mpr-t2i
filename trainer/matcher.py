# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
This comes from https://github.com/facebookresearch/detr/blob/main/models/matcher.py
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, weight_no_update=1.):
        """Creates the matcher
        Params:
        """
        super().__init__()
        self.weight_no_update = weight_no_update

    @torch.no_grad()
    def forward(self, outputs, targets, update_batch=None):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        outputs_prob = F.softmax(outputs, dim=-1)
        C = (1 - outputs_prob[:, targets.long()]).detach().cpu()
        # if outputs_prob.shape[1] > 2:
        #     C = (1 - outputs_prob[:, targets.long()]).detach().cpu()
        # else:
        #     bceloss = [F.binary_cross_entropy(outputs_prob[:, t.long()], torch.ones_like(targets), reduction="none") for t in targets]
        #     C = torch.stack(bceloss, dim=-1).detach().cpu()
        if update_batch is not None and update_batch < outputs.shape[0]:
            C[update_batch:, :] *= self.weight_no_update
        row_ind, col_ind = linear_sum_assignment(C)
        return row_ind, col_ind