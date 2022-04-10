import torch


class k_fold_generator:
    def __init__(self, k, i, features, labels):
        fold_size = features.shape[0] // k
        train_features, train_labels = None, None

        for j in range(k):
            idx = slice(j * fold_size, (j+1) * fold_size)
            if j == i:
                val_features, val_labels = features[idx, :], labels[idx]
            elif train_features is None:
                train_features, train_labels = features[idx, :], labels[idx]
            else:
                train_features = torch.cat((train_features, features[idx, :]), dim=0)
                train_labels = torch.cat((train_labels, labels[idx]), dim=0)

        self.train_features, self.train_labels = train_features, train_labels
        self.val_features, self.val_labels = val_features, val_labels
    '''
    k_fold_validator?
    '''