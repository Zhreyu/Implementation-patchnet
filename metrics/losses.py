import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils import l2_norm  

class InterLoss(nn.Module):
    def __init__(self, delta) -> None:
        super().__init__()
        self.delta = delta
        
    def forward(self, pos, neg):
        pos = l2_norm(pos, 0)
        neg = l2_norm(neg, 0)
        dot = torch.mm(pos.T, neg)
        diag = torch.ones(dot.shape[0], device=pos.device)
        dot -= torch.diag(diag)
        max_inter_dot = dot.max()
        dot = torch.mm(pos.T, pos)
        dot += torch.diag(diag*1e+2)
        min_intra_dot = dot.min()
        dot = torch.mm(neg.T, neg)
        dot += torch.diag(diag)
        if dot.min() < min_intra_dot:
            min_intra_dot = dot.min()
        loss = max_inter_dot - min_intra_dot + self.delta
        return loss if loss > 0 else 0

class IntraLoss(nn.Module):
    def __init__(self, delta) -> None:
        super().__init__()
        self.delta = delta
    
    def forward(self, pos, neg):
        loss = 0
        pos = l2_norm(pos, 0)
        neg = l2_norm(neg, 0)
        dot = torch.mm(pos.T, pos)
        dot -= self.delta
        dot = torch.triu(dot, diagonal=1)
        loss += dot.sum()
        dot = torch.mm(neg.T, neg)
        dot -= self.delta
        dot = torch.triu(dot, diagonal=1)
        loss += dot.sum()
        return loss if loss > 0 else 0
        
    
class DataLoss(nn.Module):
    def __init__(self, scale, margin) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        
    def forward(self, dist, label):
        loss = 0
        for i in range(len(label)):
            temp = torch.exp(self.scale*torch.cos(dist[label[i], i] + self.margin))
            loss += -torch.log(temp/(temp + torch.exp(self.scale*torch.cos(dist[1 - label[i], i]))))
        return loss


class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)
    

# class AdMSoftmaxLoss(nn.Module):

#     def __init__(self, in_features, out_features, s=30.0, m=0.4):
#         '''
#         AM Softmax Loss
#         '''
#         super(AdMSoftmaxLoss, self).__init__()
#         self.s = s
#         self.m = m
#         self.in_features = in_features
#         self.out_features = out_features
#         self.fc = nn.Linear(in_features, out_features, bias=False)
        
#     def forward(self, x, labels):
#         '''
#         input shape (N, in_features)
#         '''
#         assert len(x) == len(labels)
#         assert torch.min(labels) >= 0
#         assert torch.max(labels) < self.out_features
        
#         # Normalize parameters and input
#         for W in self.fc.parameters():
#             W = F.normalize(W, p=2, dim=1)
#         x = F.normalize(x, p=2, dim=1)

#         wf = self.fc(x)
#         numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
#         excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
#         denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)

#         L = numerator - torch.log(denominator)
        
#         return - torch.mean(L)

class AdMSoftmaxLoss(nn.Module):

    # def __init__(self, in_features, out_features, s=30.0, m=0.4):
    def __init__(self, in_features, out_features, s=30.0, m_l=0.4, m_s=0.1):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = [m_s, m_l]
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input: 
            x shape (N, in_features)
            labels shape (N)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W , dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        m = torch.tensor([self.m[ele] for ele in labels]).to(x.device)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        
        L = numerator - torch.log(denominator)
        
        return - torch.mean(L)



class PatchLoss(nn.Module):

    def __init__(self, alpha1=1.0, alpha2=1.0,s=30.0,m_l=0.4,m_s=0.1):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sim_loss = SimilarityLoss()
        self.amsm_loss = AdMSoftmaxLoss(512,2)
        self.s= s
        self.m_l = m_l
        self.m_s = m_s

    
    def forward(self, x1, x2, label):
        amsm_loss1 = self.amsm_loss(x1.squeeze(), label.type(torch.long).squeeze())
        amsm_loss2 = self.amsm_loss(x2.squeeze(), label.type(torch.long).squeeze())
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        sim_loss = self.sim_loss(x1, x2)
        loss = self.alpha1 * sim_loss + self.alpha2 * (amsm_loss1 + amsm_loss2)
        
        return loss
