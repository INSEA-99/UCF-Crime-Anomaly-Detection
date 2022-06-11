import torch
from torch import nn


class AnomalyDetector(nn.Module):
    def __init__(self, input_dim=4096):
        super(AnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(512, 32)
        self.dropout2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

        # In the original keras code they use "glorot_normal"
        # As I understand, this is the same as xavier normal in Pytorch
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x


def original_objective(y_pred, y_true):
    # y_pred (batch_size, 32, 1)
    # y_true (batch_size)
    lambdas = 8e-5

    normal_vids_indices = torch.where(y_true == 0)
    anomal_vids_indices = torch.where(y_true == 1)

    normal_segments_scores = y_pred[normal_vids_indices].squeeze(-1)  # (batch/2, 32, 1)
    anomal_segments_scores = y_pred[anomal_vids_indices].squeeze(-1)  # (batch/2, 32, 1)

    # get the max score for each video
    normal_segments_scores_maxes = normal_segments_scores.max(dim=-1)[0]
    anomal_segments_scores_maxes = anomal_segments_scores.max(dim=-1)[0]
    
    hinge_loss = 1 - anomal_segments_scores_maxes + normal_segments_scores_maxes
    hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

    """
    Smoothness of anomalous video
    """
    smoothed_scores = anomal_segments_scores[:, 1:] - anomal_segments_scores[:, :-1]
    smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)

    """
    Sparsity of anomalous video
    """
    sparsity_loss = anomal_segments_scores.sum(dim=-1)

    final_loss = (hinge_loss + lambdas*smoothed_scores_sum_squared + lambdas*sparsity_loss).mean()
    return final_loss


def custom_objective(y_pred, y_true):
    # y_pred (batch_size, 32, 1)
    # y_true (batch_size)
    lambdas = 8e-5

    normal_vids_indices = torch.where(y_true == 0)
    anomal_vids_indices = torch.where(y_true == 1)

    normal_segments_scores = y_pred[normal_vids_indices].squeeze(-1)  # (batch/2, 32, 1)
    anomal_segments_scores = y_pred[anomal_vids_indices].squeeze(-1)  # (batch/2, 32, 1)

    # get the max score for each video
    
    # gpu 사용시 weight 변수 cuda로 보내주기
    # weight = torch.tensor([0.5, 0.5]).cuda()

    # Using top 2 average
    weight = torch.tensor([0.5, 0.5])
    normal_segments_scores_maxes = normal_segments_scores.topk(k=2, dim=-1)[0] * weight
    anomal_segments_scores_maxes = anomal_segments_scores.topk(k=2, dim=-1)[0] * weight
    
    # Using top 3 average
    # weight = torch.tensor([1/3, 1/3, 1/3])
    # normal_segments_scores_maxes = normal_segments_scores.topk(k=3, dim=-1)[0] * weight
    # anomal_segments_scores_maxes = anomal_segments_scores.topk(k=3, dim=-1)[0] * weight

    # Using top 3 weighted average
    # weight = torch.tensor([0.5, 0.3, 0.2])
    # normal_segments_scores_maxes = normal_segments_scores.topk(k=3, dim=-1)[0] * weight
    # anomal_segments_scores_maxes = anomal_segments_scores.topk(k=3, dim=-1)[0] * weight
    
    # Using top 4 average
    # weight = torch.tensor([0.25, 0.25, 0.25, 0.25])
    # normal_segments_scores_maxes = normal_segments_scores.topk(k=4, dim=-1)[0] * weight
    # anomal_segments_scores_maxes = anomal_segments_scores.topk(k=4, dim=-1)[0] * weight

    # Using top 4 weighted average
    # weight = torch.tensor([0.4, 0.3, 0.2, 0.1])
    # normal_segments_scores_maxes = normal_segments_scores.topk(k=4, dim=-1)[0] * weight
    # anomal_segments_scores_maxes = anomal_segments_scores.topk(k=4, dim=-1)[0] * weight
    
    # # Using top 5 average
    # weight = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    # normal_segments_scores_maxes = normal_segments_scores.topk(k=5, dim=-1)[0] * weight
    # anomal_segments_scores_maxes = anomal_segments_scores.topk(k=5, dim=-1)[0] * weight
    
    # Using top 5 weighted average
    # weight = torch.tensor([0.3, 0.3, 0.2, 0.1, 0.1])
    # normal_segments_scores_maxes = normal_segments_scores.topk(k=5, dim=-1)[0] * weight
    # anomal_segments_scores_maxes = anomal_segments_scores.topk(k=5, dim=-1)[0] * weight
    
    # # # Using top 6 average
    # weight = torch.tensor([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
    # normal_segments_scores_maxes = normal_segments_scores.topk(k=6, dim=-1)[0] * weight
    # anomal_segments_scores_maxes = anomal_segments_scores.topk(k=6, dim=-1)[0] * weight
    
    # # Using top 6 average
    # weight = torch.tensor([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])
    # normal_segments_scores_maxes = normal_segments_scores.topk(k=6, dim=-1)[0] * weight
    # anomal_segments_scores_maxes = anomal_segments_scores.topk(k=6, dim=-1)[0] * weight
    
    hinge_loss = 1 - torch.sum(anomal_segments_scores_maxes, dim=-1) + torch.sum(normal_segments_scores_maxes, dim=-1)
    hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

    """
    Smoothness of anomalous video
    """
    smoothed_scores = anomal_segments_scores[:, 1:] - anomal_segments_scores[:, :-1]
    smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)

    """
    Sparsity of anomalous video
    """
    sparsity_loss = anomal_segments_scores.sum(dim=-1)

    final_loss = (hinge_loss + lambdas*smoothed_scores_sum_squared + lambdas*sparsity_loss).mean()
    return final_loss


class RegularizedLoss(torch.nn.Module):
    def __init__(self, model, original_objective, lambdas=0.001):
        super(RegularizedLoss, self).__init__()
        self.lambdas = lambdas
        self.model = model
        self.objective = original_objective

    def forward(self, y_pred, y_true):
        # loss
        # Our loss is defined with respect to l2 regularization, as used in the original keras code
        fc1_params = torch.cat(tuple([x.view(-1) for x in self.model.fc1.parameters()]))
        fc2_params = torch.cat(tuple([x.view(-1) for x in self.model.fc2.parameters()]))
        fc3_params = torch.cat(tuple([x.view(-1) for x in self.model.fc3.parameters()]))

        l1_regularization = self.lambdas * torch.norm(fc1_params, p=2)
        l2_regularization = self.lambdas * torch.norm(fc2_params, p=2)
        l3_regularization = self.lambdas * torch.norm(fc3_params, p=2)

        return self.objective(y_pred, y_true) + l1_regularization + l2_regularization + l3_regularization

