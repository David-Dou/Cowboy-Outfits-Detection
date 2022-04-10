import torch


class device_selection:
    def __init__(self, cuda_idx=0):
        """Return gpu(i) if exists, otherwise return cpu()."""
        if torch.cuda.device_count() >= cuda_idx + 1:
            self.device = torch.device(f'cuda:{cuda_idx}')
        else:
            self.device = torch.device('cpu')
