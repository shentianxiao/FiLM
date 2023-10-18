import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta


def calc_prob(logits, target_ids):
    loss = F.cross_entropy(logits.transpose(1, 2), target_ids, reduction="none")
    return (-loss).exp()


def calc_entropy(logits):
    prob = logits.softmax(dim=-1)
    logp = logits.log_softmax(dim=-1)
    return -(prob * logp).sum(dim=-1)


def sample_prob(size, func, param):
    if func == "beta":
        alpha, beta = param
        return Beta(alpha, beta).sample((size,))
    else:
        peak = param[0]
        if func == "delta":
            return torch.full((size,), peak)
        elif func == "triangular":
            p = torch.rand(size)
            f1 = torch.sqrt(peak*p)
            f2 = 1 - torch.sqrt((1-peak)*(1-p))
            return torch.where(p < peak, f1, f2)
        else:
            raise ValueError


def nucleus_sampling(logits, top_p, eps=0):
    probs = logits.softmax(dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = sorted_probs.cumsum(dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    nucleus &= sorted_probs > eps
    sorted_probs[~nucleus] = 0
    sorted_probs = sorted_probs[..., :nucleus.sum(dim=-1).max()]
    sampler = torch.distributions.categorical.Categorical(sorted_probs)
    sampled_sorted_indexes = sampler.sample()
    res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(dim=-1))
    return res.squeeze(dim=-1)


def select_pos(order, mask, empty, entropy):
    mask_f = mask.float()
    mask_f[empty, :] = 1.
    if order == "random":
        return torch.multinomial(mask_f, 1)[:, 0]
    elif order == "min-entropy":
        return torch.where(mask, entropy, float("inf")).argmin(dim=1)
    elif order == "max-entropy":
        return torch.where(mask, entropy, 0.).argmax(dim=1)
    elif order == "left2right":
        return mask_f.argmax(dim=1)
    elif order == "right2left":
        return mask_f.shape[1] - 1 - mask_f.flip(dims=[1]).argmax(dim=1)
    else:
        raise ValueError