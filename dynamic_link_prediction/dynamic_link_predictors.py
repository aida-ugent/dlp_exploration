import torch
import torch.nn as nn
from abc import abstractmethod
from pairing import SzudzikPair


class DynamicLinkPredictor(nn.Module):
    """
    Generic class for dynamic link predictors
    implements
     - __call__, which scores a batch of events according to the current state of the memory
     - memorize, which memorizes the events in the batch.
    """

    def __init__(self, temporal_data, device) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, src, dst, t, msg):
        raise NotImplementedError

    @abstractmethod
    def memorize(self, src, dst, t, msg):
        raise NotImplementedError


def encode_edge(src, dst, device="cuda"):
    return torch.tensor(
        SzudzikPair.encode(src.cpu().numpy(), dst.cpu().numpy()),
        device=device,
    )


class EdgeBankLinkPredictor(DynamicLinkPredictor):
    """
    A wrapper around edgebank, to make it work with Tensors
    """

    def __init__(self, temporal_data, device) -> None:
        super().__init__(temporal_data, device)

    def init(self):
        self.edge_keys = torch.empty(0).to(self.device)

    def __call__(self, src, dst, t, msg):
        new_edge_keys = encode_edge(src, dst)
        return torch.isin(new_edge_keys, self.edge_keys).float()

    def memorize(self, src, dst, t, msg):
        new_edge_keys = encode_edge(src, dst)
        self.edge_keys = torch.unique(torch.cat([self.edge_keys, new_edge_keys]))


class EdgeBankWindowLinkPredictor(DynamicLinkPredictor):
    def __init__(self, temporal_data, device, w=0.1) -> None:
        super().__init__(temporal_data, device)
        tmin = min(temporal_data.t)
        tmax = max(temporal_data.t)
        self.tw = (tmax - tmin) * w + tmin
        self.device = device

    def init(self):
        self.e_hist = torch.empty(0).to(self.device)
        self.t_hist = torch.empty(0).to(self.device)

    def __call__(self, src, dst, t, msg):
        min_t = t.min()
        start_t = min_t - self.tw
        end_t = min_t
        mask = (self.t_hist >= start_t) & (self.t_hist <= end_t)
        edge_hist = self.e_hist[mask]
        edge_keys = encode_edge(src, dst)
        return torch.isin(edge_keys, edge_hist).float()

    def memorize(self, src, dst, t, msg):
        new_edge_keys = encode_edge(src, dst)
        self.e_hist = torch.cat([self.e_hist, new_edge_keys])
        self.t_hist = torch.cat([self.t_hist, t])


class PreferentialAttachmentLinkPredictor(DynamicLinkPredictor):
    def __init__(self, temporal_data, device) -> None:
        super().__init__(temporal_data, device)

    def __call__(self, src, dst, t, msg):
        src_obs = torch.isin(src, self.src_hist).float()
        dst_obs = torch.isin(dst, self.dst_hist).float()
        return src_obs * dst_obs

    def memorize(self, src, dst, t, msg):
        self.src_hist = torch.unique(torch.cat([self.src_hist, src]))
        self.dst_hist = torch.unique(torch.cat([self.dst_hist, dst]))

    def init(self):
        self.src_hist = torch.empty(0).to(self.device)
        self.dst_hist = torch.empty(0).to(self.device)
