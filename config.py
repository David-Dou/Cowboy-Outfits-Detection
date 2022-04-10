import warnings


class Config(object):
    # model = ''
    data_dir = 'data/cowboyoutfits'  # directory relative to main.py
    lr = 0.005
    momentum = 0.9  # ?
    weight_decay = 0.0005

    batch_size = 2
    num_epochs = 10
    model_save_freq = 5
    load_model_path = None

    use_gpu = True
    num_workers = 0

    def parse(self, kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                warnings.warn('Warning: opt has no attribute %s' % key)
            setattr(self, key, value)
