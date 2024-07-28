import torch


def batch_index(src, indices, batch_first=True):
    if batch_first:
        # src: b, n, c
        # indices: b, m
        # tgt: b, m, c
        b, m = indices.shape[:2]
        batch_indices = torch.arange(b, device=src.device).unsqueeze(-1).repeat(1, m)
        return src[batch_indices, indices]
    else:
        # src: n, b, c
        # indices: m, b
        # tgt: m, b, c
        m, b = indices.shape[:2]
        batch_indices = torch.arange(b, device=src.device).unsqueeze(0).repeat(m, 1)
        return src[indices, batch_indices]


def batch_scatter(src, indices, content, batch_first=True):
    if batch_first:
        # src: b, n, c
        # indices: b, m
        # tgt: b, m, c
        b, m = indices.shape[:2]
        batch_indices = torch.arange(b, device=src.device).unsqueeze(-1).repeat(1, m)
        src[batch_indices, indices] = content
    else:
        # src: n, b, c
        # indices: m, b
        # tgt: m, b, c
        m, b = indices.shape[:2]
        batch_indices = torch.arange(b, device=src.device).unsqueeze(0).repeat(m, 1)
        src[indices, batch_indices] = content

    return src
