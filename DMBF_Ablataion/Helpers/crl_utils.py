import numpy as np
import torch
import torch.nn.functional as F

# correctness history class
class History(object):
    def __init__(self, n_data):
        self.correctness = np.zeros((n_data))
        self.confidence = np.zeros((n_data))
        self.max_correctness = 1

    # correctness update
    def correctness_update(self, data_idx, correctness, output):
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()

        self.correctness[data_idx] += correctness.cpu().numpy()
        self.confidence[data_idx] = confidence.cpu().detach().numpy()

    # max correctness update
    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data):
        data_min = self.correctness.min()
        data_max = float(self.max_correctness)

        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, data_idx1, data_idx2, device):
        data_idx1 = data_idx1.cpu().numpy()
        cum_correctness1 = self.correctness[data_idx1] # 배치 내 x_i-1 샘플의 correctness
        cum_correctness2 = self.correctness[data_idx2] # 배치 내 x_i 샘플의 correctness
        # normalize correctness values
        cum_correctness1 = self.correctness_normalize(cum_correctness1)
        cum_correctness2 = self.correctness_normalize(cum_correctness2)
        # make target pair
        n_pair = len(data_idx1)
        target1 = cum_correctness1[:n_pair]
        target2 = cum_correctness2[:n_pair]
        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.from_numpy(target).float().to(device)
        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().to(device)

        return target, margin
    
# rank target entropy
def negative_entropy(data, normalize=False, max_value=None):
    softmax = F.softmax(data, dim=1)
    log_softmax = F.log_softmax(data, dim=1)
    entropy = softmax * log_softmax
    entropy = -1.0 * entropy.sum(dim=1)
    # normalize [0 ~ 1]
    if normalize:
        normalized_entropy = entropy / max_value
        return -normalized_entropy

    return -entropy
