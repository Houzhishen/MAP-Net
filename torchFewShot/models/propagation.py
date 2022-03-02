import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embeddings import EmbeddingWord


class MLP(nn.Module):
    def __init__(self, indim, drop_rate):
        super(MLP, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Linear(indim, 300),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Linear(300, 1)
                        )
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)

        out = F.sigmoid(out)

        return out


class RelationModify(nn.Module):
    def __init__(self, indim, scale=2, drop_rate=0.0):
        super(RelationModify, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Linear(indim, int(indim/scale)),
                        nn.LeakyReLU())
        self.layer2 = nn.Sequential(
                        nn.Linear(int(indim/scale), indim)
                        )
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)


        return out


class Propagation(nn.Module):
    """Label Propagation"""
    def __init__(self, args, word_dim, out_channels):
        super(Propagation, self).__init__()

        self.args = args
        self.wenc_module = EmbeddingWord(word_dim, out_channels, args.drop_rate)
        self.mlp = MLP(out_channels*2, args.drop_rate)
        self.relation_modify = RelationModify(out_channels)
        self.mse = nn.MSELoss()

        self.batch_size = self.args.train_batch
        self.num_supports = self.args.nKnovel * self.args.nExemplars
        self.num_queries = self.args.train_nTestNovel
        self.num_samples = self.num_supports + self.num_queries

    def forward(self, full_images, full_words, full_words_real):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """

        batch_size, num_samples, feat_dim = full_images.size()

        # support_images = full_images[:, :self.num_supports]

        support_words = full_words[:, :self.num_supports].contiguous()
        support_word_embeddings_view = self.wenc_module(support_words.view(-1, *support_words.shape[2:]))
        support_word_embeddings = support_word_embeddings_view.view(*support_words.shape[:2], -1)

        query_word_embeddings = torch.zeros(self.batch_size*self.num_queries, 1, support_word_embeddings.shape[-1]).cuda()
        full_word_embeddings = torch.cat([support_word_embeddings, query_word_embeddings], dim=1)

        Ei, addition_loss = self.get_similiar(full_images, full_word_embeddings)
        Si = self.normalize(Ei)

        I = torch.eye(num_samples).unsqueeze(0).repeat(self.batch_size*self.num_queries, 1, 1).cuda()
        Pi = (1 - self.args.alpha) * torch.inverse(I - self.args.alpha * Si)
        
        full_images = torch.bmm(Pi, full_images)
        full_word_embeddings = torch.bmm(Pi, full_word_embeddings)

        full_data = torch.cat([full_images, full_word_embeddings], dim=-1)
        full_data_view = full_data.view(-1, *full_data.shape[2:])
        sigma = self.mlp(full_data_view).view(*full_data.shape[:2], -1)

        full_data = sigma * full_images + (1-sigma) * full_word_embeddings

        return full_data, addition_loss

    def get_similiar(self, x, y):
        
        mask = torch.ones(self.num_supports, self.num_supports).cuda()
        mask = torch.triu(mask, 1).view(-1)
        index = mask.nonzero().view(-1)
        
        x_view = x.view(self.batch_size, self.num_queries, self.num_supports+1, -1)
        x_support = x_view[:, 0, :-1, :]
        x_support_i = x_support.unsqueeze(2)
        x_support_j = x_support_i.transpose(1, 2)
        x_support_ij = torch.pow(x_support_i - x_support_j, 2)
        x_support_ij = x_support_ij.contiguous().view(self.batch_size, -1, x.shape[-1])   # (batch*nums*nums, dim)
        x_support_ij = x_support_ij[:, index].contiguous().view(-1, x.shape[-1])
        
        y_view = y.view(self.batch_size, self.num_queries, self.num_supports+1, -1)
        y_support = y_view[:, 0, :-1, :]
        y_support_i = y_support.unsqueeze(2)
        y_support_j = y_support_i.transpose(1, 2)
        y_support_ij = torch.pow(y_support_i - y_support_j, 2)
        y_support_ij = y_support_ij.contiguous().view(self.batch_size, -1, y.shape[-1])  # (batch*nums*nums, dim)
        y_support_ij = y_support_ij[:, index].contiguous().view(-1, y.shape[-1])

        x_support_ij_modify = self.relation_modify(x_support_ij)
        loss_modify = F.mse_loss(x_support_ij_modify, y_support_ij)

        x_i = x.unsqueeze(2)
        x_j = x_i.transpose(1, 2)
        x_ij = torch.pow(x_i - x_j, 2)
        x_ij = x_ij.view(-1, x.shape[-1])
        
        y_ij = self.relation_modify(x_ij)

        y_ij = y_ij.view(x.shape[0], x.shape[1], x.shape[1], -1)
        y_ij = y_ij.mean(-1)
        mask = y_ij != 0
        sigma = (y_ij*mask).std(dim=(1, 2), keepdim=True)
        y_ij = torch.exp(-y_ij / sigma)
        diag_mask = 1.0 - torch.eye(y.size(1)).unsqueeze(0).repeat(y.size(0), 1, 1).cuda()
        y_ij = y_ij * diag_mask

        return y_ij, loss_modify * self.args.miu

    def normalize(self, W):
        
        D       = W.sum(1)
        D_sqrt_inv = torch.sqrt(1.0/(D + 1e-6))
        D1      = torch.unsqueeze(D_sqrt_inv,2).repeat(1, 1, W.size(1))
        D2      = torch.unsqueeze(D_sqrt_inv,1).repeat(1, W.size(1), 1)
        S       = D1*W*D2

        return S

