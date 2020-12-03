import torch
import torch.nn as nn


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: if true, output is the pairwise squared euclidean
            distance matrix. If false, output is the pairwise euclidean
            distance matrix
    Returns:
        torch.Tensor: pairwise matrix of size (batch_size, batch_size)
    """
    # Get squared L2 norm for each embedding.
    # We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability
    # (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square = torch.mm(embeddings, embeddings.t())
    diag = torch.diag(square)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = diag.view(-1, 1) - 2.0 * square + diag.view(1, -1)

    # Because of computation errors, some distances
    # might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite
        # when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 - mask) * torch.sqrt(distances)

    return distances


def pairwise_distance_torch(embeddings: torch.Tensor):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(
        pairwise_distances_squared,
        torch.tensor([0.]).to(pairwise_distances_squared.device)
    )
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones(
        (pairwise_distances.shape[0], pairwise_distances.shape[1])
    ) - torch.diag(torch.ones(pairwise_distances.shape[0]))

    pairwise_distances = torch.mul(
        pairwise_distances,
        mask_offdiagonals.to(device=pairwise_distances.device)
    )
    return pairwise_distances


class TripletSemiHardLoss(nn.Module):

    def __init__(self, margin=0.1):
        super().__init__()

        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Computes the triplet loss_functions with semi-hard negative mining.
           The loss_functions encourages the positive distances (between a pair of embeddings
           with the same labels) to be smaller than the minimum negative distance
           among which are at least greater than the positive distance plus the
           margin constant (called semi-hard negative) in the mini-batch.
           If no such negative exists, uses the largest negative distance instead.
           See: https://arxiv.org/abs/1503.03832.
           We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
           [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
           2-D float `Tensor` of l2 normalized embedding vectors.
           Args:
             embeddings:
             labels:
           """

        # Reshape label tensor to [batch_size, 1].
        lshape = labels.shape
        labels = torch.reshape(labels, [lshape[0], 1])

        pdist_matrix = pairwise_distance_torch(embeddings)

        # Build pairwise binary adjacency matrix.
        adjacency = torch.eq(labels, labels.transpose(0, 1))
        # Invert so we can select negatives only.
        adjacency_not = adjacency.logical_not()

        batch_size = labels.shape[0]

        # Compute the mask.
        pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
        adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

        transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
        greater = pdist_matrix_tile > transpose_reshape

        mask = adjacency_not_tile & greater

        # final mask
        mask_step = mask.to(dtype=torch.float32)
        mask_step = mask_step.sum(axis=1)
        mask_step = mask_step > 0.0
        mask_final = mask_step.reshape(batch_size, batch_size)
        mask_final = mask_final.transpose(0, 1)

        adjacency_not = adjacency_not.to(dtype=torch.float32)
        mask = mask.to(dtype=torch.float32)

        # negatives_outside: smallest D_an where D_an > D_ap.
        axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
        masked_minimums = \
        torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1,
                  keepdim=True)[0] + axis_maximums[0]
        negatives_outside = masked_minimums.reshape([batch_size, batch_size])
        negatives_outside = negatives_outside.transpose(0, 1)

        # negatives_inside: largest D_an.
        axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
        masked_maximums = \
        torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not),
                  dim=1, keepdim=True)[0] + axis_minimums[0]
        negatives_inside = masked_maximums.repeat(1, batch_size)

        semi_hard_negatives = torch.where(mask_final, negatives_outside,
                                          negatives_inside)

        loss_mat = self.margin + pdist_matrix - semi_hard_negatives

        mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(
            torch.ones(batch_size)).to(self.device)
        num_positives = mask_positives.sum()

        triplet_loss = (torch.max(
            torch.mul(loss_mat, mask_positives),
            torch.tensor([0.]).to(self.device)
        )).sum() / num_positives

        triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
        return triplet_loss
