import utils


class Trainer:
    '''
    parameters with default values should be placed after those without defaults values
    '''

    def __init__(self, data_tensors, batch_size, num_workers, num_epoch, model, loss, optimizer):
        iter_generator = utils.iter_generator(data_tensors,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              is_train=True)
        self.train_iter = iter_generator.data_iter
        self.num_epoch = num_epoch
        self.loss = loss
        self.optimizer = optimizer
        self.model = model

        device_selector = utils.device_selection()
        self.device = device_selector.device

    def train(self):
        for epoch in range(self.num_epoch):
            for batch_features, batch_labels in self.train_iter:
                self.optimizer.zero_grad()
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                l = self.loss(self.model(batch_features), batch_labels)
                l.backward()
                self.optimizer.step()
