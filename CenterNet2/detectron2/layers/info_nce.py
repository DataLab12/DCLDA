import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """


    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', positive_mode='single'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.positive_mode = positive_mode

    def forward(self, query, positive_keys, negative_keys=None):
        return info_nce(query, positive_keys, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode, positive_mode= self.positive_mode)


def info_nce(query, positive_keys, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired' , positive_mode='single'):
    # Check input dimensionality.

    # print(f"Check shape: Q: {query.shape}, P: {positive_keys.shape}, N: {negative_keys.shape}")
    # print(f"Check grad: Q: {query.grad_fn}, P: {positive_keys.grad_fn}, N: {negative_keys.grad_fn}")
    # print("pos mode: ",positive_mode)

    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_keys.dim() != 2:
        raise ValueError('<positive_keys> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if positive_mode == 'single' and len(query) != len(positive_keys):
        raise ValueError('<query> and <positive_keys> must must have the same number of samples.')
    if positive_mode == 'multiple' and len(positive_keys) <= 1:
        raise ValueError('<positive_keys> must must have more than one samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_keys.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_keys> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_keys, negative_keys = normalize(query, positive_keys, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        if positive_mode == 'multiple':
            positive_logits = query @ transpose(positive_keys)
            positive_logits = torch.mean(positive_logits, dim=1,  keepdim=True)
        else:
            positive_logits = torch.sum(query * positive_keys, dim=1, keepdim=True)

        # print("Pos logit: ",positive_logits.shape)
        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)
            # print("neg logit: ",negative_logits.shape)
        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_keys)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)



#     def __init__(self, temperature= 0.12, reduction='mean', negative_mode='unpaired'):
#         super().__init__()
#         self.temperature = temperature
#         self.reduction = reduction
#         self.negative_mode = negative_mode

#     def forward(self, query, positive_key, negative_keys=None):
#         return info_nce(query, positive_key, negative_keys,
#                         temperature=self.temperature,
#                         reduction=self.reduction,
#                         negative_mode=self.negative_mode)


# def info_nce(query, positive_key, negative_keys=None, temperature= 0.12, reduction='mean', negative_mode='unpaired'):

#     positive_key = positive_key.detach()
#     negative_keys = negative_keys.detach()
#     # print(f"Shape of query: {query.shape}")
#     # print(f"Shape of positive: {positive_key.shape}")
#     # print(f"Shape of negatives: {negative_keys.shape}")
#     # print(f"Temparature: {temperature}")
#     # print(f"Check device: Q: {query.is_cuda}, P: {positive_key.is_cuda}, N: {negative_keys.is_cuda}")
#     # print(f"Check grad: Q: {query.grad_fn}, P: {positive_key.grad_fn}, N: {negative_keys.grad_fn}")

#     # Check input dimensionality.
#     if query.dim() != 2:
#         raise ValueError('<query> must have 2 dimensions.')
#     if positive_key.dim() != 2:
#         raise ValueError('<positive_key> must have 2 dimensions.')
#     if negative_keys is not None:
#         if negative_mode == 'unpaired' and negative_keys.dim() != 2:
#             raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
#         if negative_mode == 'paired' and negative_keys.dim() != 3:
#             raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

#     # Check matching number of samples.
#     if len(query) != len(positive_key):
#         raise ValueError('<query> and <positive_key> must must have the same number of samples.')
#     if negative_keys is not None:
#         if negative_mode == 'paired' and len(query) != len(negative_keys):
#             raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

#     # Embedding vectors should have same number of components.
#     if query.shape[-1] != positive_key.shape[-1]:
#         raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
#     if negative_keys is not None:
#         if query.shape[-1] != negative_keys.shape[-1]:
#             raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

#     # Normalize to unit vectors
#     query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

#     if negative_keys is not None:
#         # Explicit negative keys

#         # Cosine between positive pairs
#         positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

#         if negative_mode == 'unpaired':
#             # Cosine between all query-negative combinations
#             negative_logits = query @ transpose(negative_keys)

#         elif negative_mode == 'paired':
#             query = query.unsqueeze(1)
#             negative_logits = query @ transpose(negative_keys)
#             negative_logits = negative_logits.squeeze(1)

#         # First index in last dimension are the positive samples
#         # print("before negatives: ",negative_logits)
#         # negative_logits[0][0]= 0.40
#         # print("after negatives: ",negative_logits)

        
#         # neg_samples = negative_logits.shape[1]
#         # print(f"positive_logits: {positive_logit}")
#         # print(f"######negative_logits before#######: {negative_logits}")


#         # mask= negative_logits <= 0.7 * positive_logit[0][0]
#         # negative_logits = negative_logits[mask].reshape(1,-1)
        
#         # print(f"######negative_logits after#######: {negative_logits}") 

#         logits = torch.cat([positive_logit, negative_logits], dim=1)
#         labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

#         # if int(logits.shape[1]) != (neg_samples+1):
#         #     print("False Negative Found!!!!")
#             # exit()
       
#         # print(f"Shape of logits: {logits.shape}")
#     else:
#         # Negative keys are implicitly off-diagonal positive keys.

#         # Cosine between all combinations
#         logits = query @ transpose(positive_key)

#         # Positive keys are the entries on the diagonal
#         labels = torch.arange(len(query), device=query.device)

#     return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
