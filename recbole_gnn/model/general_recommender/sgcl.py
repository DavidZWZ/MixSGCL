# -*- coding: utf-8 -*-
r"""
SGCL
################################################
"""
import torch
import torch.nn.functional as F
from recbole_gnn.model.general_recommender import LightGCN
from recbole.utils import InputType

class SGCL(LightGCN):
    def __init__(self, config, dataset):
        super(SGCL, self).__init__(config, dataset)
        self.temperature = config['temperature']
        
    def forward(self):
        all_embs = self.get_ego_embeddings()
        embeddings_list = [all_embs]

        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embs)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
    
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[pos_item]

        u_embeddings, i_embeddings = F.normalize(u_embeddings, dim=-1), F.normalize(i_embeddings, dim=-1)
        pos_score = (u_embeddings * i_embeddings).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)

        ttl_u_score = torch.matmul(u_embeddings, u_embeddings.transpose(0, 1)) 
        ttl_u_score = torch.exp(ttl_u_score / self.temperature).sum(dim=1)
        ttl_i_score = torch.matmul(i_embeddings, i_embeddings.transpose(0, 1))
        ttl_i_score = torch.exp(ttl_i_score / self.temperature).sum(dim=1)

        ttl_score = ttl_u_score + ttl_i_score
        sup_cl_loss = -torch.log(pos_score / ttl_score).sum()

        return sup_cl_loss
    


