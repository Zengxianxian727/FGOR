import torch
from tqdm import tqdm
import numpy as np
import sys

def DGLoss(W):
    W_1, W_2 = W.size()
    total = W_1**2 - W_1
    W_mm = torch.mm(W, W.t()) * (torch.ones(W_1) - torch.eye(W_1)).cuda()
    loss = torch.sum(torch.abs(W_mm).view(-1)) / total
    return loss

def train_2(model, train_loader, criterion, optimizer, Lambda=0, Lambda2 = 0):

    #if criterion_begin==None:
    #    criterion_begin = criterion

    print('Training...')

    epoch_loss = []
    num_correct = 0
    num_total = 0

    model.train()

    for i, (images, target) in enumerate(tqdm(train_loader)):

        image_var = torch.tensor(images).cuda()
        label = torch.tensor(target).cuda(non_blocking=True)

        y_pred, W, feat = model(image_var, train_flag=True)
        loss = criterion(y_pred, label) + Lambda * DGLoss(W)
        epoch_loss.append(loss.item())

        # Prediction
        _, prediction = torch.max(y_pred.data, 1)
        num_total += y_pred.size(0)
        num_correct += torch.sum(prediction == label.data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        #if np.isnan(loss.detach().cpu().numpy()):
        #    sys.exit('Loss diverged')

    num_correct = torch.tensor(num_correct).float().cuda()
    num_total = torch.tensor(num_total).float().cuda()

    train_acc = 100 * num_correct / num_total

    return train_acc, sum(epoch_loss) / len(epoch_loss)

class RetricMetric(object):
    def __init__(self, feats_test, labels_test):
        self.feats_test = self.feats_query = feats_test
        self.labels_test = self.labels_query = labels_test
        self.sim_mat = np.matmul(self.feats_query, np.transpose(self.feats_test))

    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0
        for i in range(m):
            pos_sim = self.sim_mat[i][self.labels_test == self.labels_query[i]]
            neg_sim = self.sim_mat[i][self.labels_test != self.labels_query[i]]

            thresh = np.sort(pos_sim)[-2] 
            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m

    def precision_at_k(self, k):
        m = len(self.sim_mat)
        accuracy = 0
        for i in range(m):
            knn_i = self.sim_mat[i]
            knn_i_index = np.argsort(-1 * knn_i)
            knn_i_labels_k = self.labels_test[knn_i_index[1:k+1]]
            accuracy_per_sample = np.sum(knn_i_labels_k == self.labels_query[i])/k
            accuracy += accuracy_per_sample

        return accuracy / m

    def mean_average_precision_at_r(self, k):
        m = len(self.sim_mat)

        k_linspace = np.linspace(1, k, k)
        all_AP = 0

        for i in range(m):
            knn_i = self.sim_mat[i]
            knn_i_index = np.argsort(-1 * knn_i)
            knn_i_labels_k = self.labels_test[knn_i_index[1:k+1]]
            equality = knn_i_labels_k == self.labels_query[i]
            num_same = np.sum(equality)
            if num_same != 0:
                AP = np.sum(np.cumsum(equality) * equality / k_linspace) / num_same
                all_AP += AP
        return all_AP / m

def test_3(model, test_loader, recall=[1], precision=[10], mAP=[100]):

    model.eval()

    epoch_loss = []

    num_total = 0

    for i, (images, target) in enumerate(tqdm(test_loader)):
        # Data.
        image_var = torch.tensor(images).cuda()
        label = torch.tensor(target).cuda(non_blocking=True)
        # Prediction.
        y_pred = model(image_var)
        if i == 0:
            all_predict_test = y_pred.data.cpu().numpy()
            all_label_test = label.data.cpu().numpy()
            #break
        else:
            all_predict_test = np.concatenate([all_predict_test, y_pred.data.cpu().numpy()], 0)
            all_label_test = np.concatenate([all_label_test, label.data.cpu().numpy()], 0)

        num_total += y_pred.size(0)

    retmatric = RetricMetric(all_predict_test, all_label_test)

    recall_output = {}
    for k in recall:
        recall_output['%d' % k] = retmatric.recall_k(k) * 100
    precision_output = {}
    for k in precision:
        precision_output['%d' % k] = retmatric.precision_at_k(k) * 100
    mAP_output = {}
    for k in mAP:
        mAP_output['%d' % k] = retmatric.mean_average_precision_at_r(k) * 100
    all_predict = all_predict_test

    return recall_output, precision_output, mAP_output, all_predict

