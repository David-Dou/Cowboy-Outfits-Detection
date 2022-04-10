from torch.utils import data


class iter_generator:
    '''
    parameters with default values should be placed after those without defaults values
    can merge train and test?
    '''

    def __init__(self, data_tensors, batch_size, num_workers, is_train):
        self.data_tensors = data_tensors
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_train = is_train

        dataset = data.TensorDataset(*data_tensors)
        self.data_iter = data.DataLoader(dataset,
                                         batch_size,
                                         num_workers=num_workers,
                                         shuffle=is_train)

    def test_iter(self):
        dataset = data.TensorDataset(self.data_tensors)
        data_iter = data.DataLoader(dataset,
                                    self.batch_size,
                                    num_workers=self.num_workers,
                                    shuffle=self.is_train)

        return data_iter