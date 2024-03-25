import numpy as np
import torch


def szudzik_encode(x, y):
    out1 = x * x + x + y
    out2 = x + y * y
    # mask = x == torch.max([x, y], dim=0)
    mask = x == torch.maximum(x, y)
    out = out1 * mask + out2 * (~mask)
    return out


def szudzik_decode(z):
    sqrtz = torch.floor(torch.sqrt(z))
    sqz = sqrtz * sqrtz
    diff = z - sqz
    mask = diff < sqrtz
    x = diff * mask + sqrtz * ~mask
    y = sqrtz * mask + (diff - sqrtz) * ~mask
    return torch.stack([x, y]).int()


class SzudzikPair:
    """
    Szudzik's pairing function
    Allows to map a pair of integers to a unique integer, in a reversible way.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def encode(x, y):
        out1 = x * x + x + y
        out2 = x + y * y
        mask = x == np.max([x, y], axis=0)
        out = out1 * mask + out2 * (1 - mask)
        return out

    @staticmethod
    def decode(z):
        sqrtz = np.floor(np.sqrt(z))
        sqz = sqrtz * sqrtz
        diff = z - sqz
        mask = diff < sqrtz
        x = diff * mask + sqrtz * ~mask
        y = sqrtz * mask + (diff - sqrtz) * ~mask
        return np.stack([x, y]).astype(int)


class OrdinalPair:
    def __init__(self) -> None:
        pass

    @staticmethod
    def encode(x, y):
        _, idx = np.unique(np.stack([x, y]), axis=0, return_inverse=True)
        return idx


class LinearIndexing:
    def __init__(self, n, m=None):
        self.n = n
        if m is None:
            self.m = n
        else:
            self.m = m

    def encode(self, x, y):
        return np.ravel_multi_index((x, y), (self.n, self.m))

    def decode(self, z):
        return np.unravel_index(z, (self.n, self.m))
