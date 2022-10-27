import torch.nn as nn
from multimodel_hierar import MultiModel_hierar
from my_utils.utils import *


class Model(nn.Module):
    def __init__(self, config, params):
        super(Model, self).__init__()
        self.config = config
        # self.use_poi = config['model']['use_aoi']
        self.method = config['model']['method'].split('_')[-1]
        self.seq_len = config['model']['max_len']
        self.device = get_device(config)
        print(f'device: {self.device}')
        self.params = params

        self.gps_embedding = nn.Linear(2, params['gps_embedding_size'])
        self.user_embedding = nn.Embedding(params['max_courier'] + 1, params['user_embedding_size'])
        self.weekday_embedding = nn.Embedding(8, params['weekday_embedding_size'])
        self.aoi_embedding = nn.Embedding(params['max_aoi'] + 1, params['aoi_embedding_size'])
        self.aoi_type_embedding = nn.Embedding(params['max_aoi_type'] + 1, params['aoi_type_embedding_size'])

        self.conti_fea_size = 16
        self.global_conti_embedding = nn.Linear(3, self.conti_fea_size)
        self.unpick_conti_embedding = nn.Linear(6, self.conti_fea_size)
        self.aoi_conti_embedding = nn.Linear(params['aoi_conti_size'], self.conti_fea_size)

        self.use_dipan = params['use_dipan']
        self.global_size = self.conti_fea_size + params['user_embedding_size'] + \
                               params['weekday_embedding_size']
        if self.use_dipan:
            self.dipan_embedding = nn.Embedding(params['max_dipan'] + 1, params['dipan_embedding_size'])
            self.global_size += params['dipan_embedding_size']

        self.courier_size = params['gps_embedding_size'] * 2 + params['aoi_type_embedding_size'] + \
                                params['aoi_embedding_size']
        self.unpick_size = self.conti_fea_size + params['gps_embedding_size'] + params['aoi_type_embedding_size'] + \
                           params['aoi_embedding_size'] + self.global_size
        self.edge_size = 5
        self.last_size = 8

        self.aoi_fea_size = self.conti_fea_size + params['gps_embedding_size'] + params['aoi_type_embedding_size'] + \
                            params['aoi_embedding_size'] + self.global_size
        self.aoi_edge_size = 3

        unpick_input_size = self.unpick_size
        self.model = MultiModel_hierar(params, self.last_size, unpick_input_size, self.edge_size,
                                       self.aoi_fea_size, self.aoi_edge_size, self.courier_size, self.device)

        self.model_name = self.model.model_name
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, *data, is_train=True, label_eta=None, aoi_eta=None):
        unpick_fea, edge_fea, unpick_len, last_fea, last_len, global_fea, idx, pos, \
            aoi_index, aoi_fea, aoi_edge, aoi_len, aoi_idx, aoi_pos = data

        user_embed = self.user_embedding(global_fea[:, 0].long())
        weekday_embed = self.weekday_embedding(global_fea[:, 3].long())

        global_conti_fea = torch.cat([global_fea[:, 1:3], global_fea[:, 4:5]], dim=1)
        global_conti_fea = self.global_conti_embedding(global_conti_fea)

        global_fea_new = torch.cat([global_conti_fea, user_embed, weekday_embed], dim=1)
        dipan_embed = self.dipan_embedding(global_fea[:, 11].long())
        global_fea_new = torch.cat([global_fea_new, dipan_embed], dim=1)

        now_gps_embed = self.gps_embedding(global_fea[:, 5:7])
        now_aoi_gps_embed = self.gps_embedding(global_fea[:, 9:11])

        now_aoi_id_embed = self.aoi_embedding(global_fea[:, 7].long())
        now_aoi_type_embed = self.aoi_type_embedding(global_fea[:, 8].long())

        courier_fea = torch.cat([now_gps_embed, now_aoi_gps_embed,
                                 now_aoi_id_embed, now_aoi_type_embed], dim=1)

        unpick_aoi_embed = self.aoi_embedding(unpick_fea[:, :, 2].long())
        unpick_aoi_type_embed = self.aoi_type_embedding(unpick_fea[:, :, 3].long())

        unpick_gps_embed = self.gps_embedding(unpick_fea[:, :, 0:2])
        unpick_conti_embed = self.unpick_conti_embedding(unpick_fea[:, :, 4:])
        unpick_fea_new = torch.cat([unpick_gps_embed, unpick_aoi_embed, unpick_aoi_type_embed, unpick_conti_embed], dim=2)

        aoi_id_embed = self.aoi_embedding(aoi_fea[:, :, 0].long())
        aoi_tpye_embed = self.aoi_type_embedding(aoi_fea[:, :, 1].long())

        aoi_gps_embed = self.gps_embedding(aoi_fea[:, :, 2:4])
        aoi_conti_embed = self.aoi_conti_embedding(aoi_fea[:, :, 4:])
        aoi_fea = torch.cat([aoi_gps_embed, aoi_id_embed, aoi_tpye_embed, aoi_conti_embed], dim=2)

        return self.model(unpick_fea_new, edge_fea, unpick_len, courier_fea, global_fea_new, idx, pos,
                          aoi_index, aoi_fea, aoi_edge, aoi_len, aoi_idx, aoi_pos,
                          is_train, label_eta, aoi_eta)


if __name__ == '__main__':
    pass
