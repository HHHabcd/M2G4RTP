import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pointer_decoder
import gat_encoder
import sortlstm_decoder


def get_init_mask(max_seq_len, batch_size, sort_len):
    """
    Get the init mask for decoder
    """
    range_tensor = torch.arange(max_seq_len, device=sort_len.device, dtype=sort_len.dtype).expand(batch_size,
                                                                                                  max_seq_len)
    each_len_tensor = sort_len.view(-1, 1).expand(batch_size, max_seq_len)
    raw_mask_tensor = range_tensor >= each_len_tensor
    return raw_mask_tensor


class PN_decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, courier_dim, max_seq_len):
        super(PN_decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.first_input_layer = nn.Linear(courier_dim, embedding_dim)
        self.decoder = pointer_decoder.Decoder(embedding_dim,
                                               hidden_dim,
                                               tanh_exploration=10,
                                               use_tanh=True,
                                               n_glimpses=1,
                                               mask_glimpses=True,
                                               courier_size=courier_dim,
                                               mask_logits=True)

    def forward(self, hidden_state, courier_fea, seq_len):
        batch = hidden_state.shape[0]
        h = torch.zeros([batch, self.hidden_dim], device=hidden_state.device)
        c = torch.zeros([batch, self.hidden_dim], device=hidden_state.device)
        init_input = self.first_input_layer(courier_fea)
        init_mask = get_init_mask(self.max_seq_len, batch, seq_len)
        hidden_state = hidden_state.permute(1, 0, 2)
        score, arg = self.decoder(decoder_input=init_input,
                                  embedded_inputs=hidden_state,
                                  hidden=(h, c),
                                  context=hidden_state,
                                  init_mask=init_mask,
                                  courier_fea=courier_fea)
        score = score.exp()
        return score, arg


class MultiModel_hierar(nn.Module):
    def __init__(self, params, last_size, unpick_size, edge_size=0,
                 aoi_size=0, aoi_edge_size=0, courier_size=0, device='cpu'):
        super(MultiModel_hierar, self).__init__()

        self.last_size = last_size
        self.unpick_size = unpick_size
        self.aoi_size = aoi_size

        self.is_pe = params['is_pe']
        self.pred_eta = params['pred_eta']
        self.pred_order = params['pred_order']
        self.hierarchical = params['hierarchical']
        self.sort_mode = params['sort_mode']
        self.order_info = params['order_info']
        self.order_detach = params['order_detach']
        self.order_aoi_fea_use = params['order_aoi_fea_use']
        self.eta_aoi_fea_use = params['eta_aoi_fea_use']
        if not self.hierarchical:
            self.order_aoi_fea_use, self.eta_aoi_fea_use = [], []

        if not self.pred_order:
            params['order_decoder'] = 'none'
            assert params['eta_decoder'] == 'mlp'
            if 'order' in self.eta_aoi_fea_use:
                self.eta_aoi_fea_use.remove('order')

        if not self.pred_eta:
            params['eta_decoder'] = 'none'
            if 'eta' in self.order_aoi_fea_use:
                self.order_aoi_fea_use.remove('eta')

        self.input_layer_hidden = params['input_hidden']
        self.input_layer_hidden_edge = params['input_edge_hidden']
        self.input_layer_hidden_aoi = params['input_aoi_hidden']
        self.input_layer_hidden_aoi_edge = params['input_aoi_edge_hidden']
        self.encoder_hidden = params['encoder_hidden_0']
        self.encoder_hidden_aoi = params['encoder_hidden_1']
        self.package_represent_size = 0
        self.aoi_represent_size = 0
        self.eta_decoder_hidden = params['eta_decoder_hidden_0']
        self.eta_decoder_hidden_aoi = params['eta_decoder_hidden_1']
        self.order_decoder_hidden = params['order_decoder_hidden_0']
        self.order_decoder_hidden_aoi = params['order_decoder_hidden_1']

        self.max_seq_len = 20
        self.max_aoi_len = 10
        self.device = device
        self.train_mode = params['train_mode']
        self.params = params

        if self.pred_order:
            self.pe_size = int(self.input_layer_hidden / 2)
            self.pe_tabel = self.positional_encoding(self.pe_size).to(device)

        self.model_name = self.get_model_name(params)
        print(self.model_name)

        self.input_layer_unpick = nn.Linear(self.unpick_size, self.input_layer_hidden)
        if self.hierarchical:
            self.input_layer_aoi = nn.Linear(self.aoi_size, self.input_layer_hidden_aoi)

        self.package_represent_size = self.encoder_hidden
        self.input_layer_edge = nn.Linear(edge_size, self.input_layer_hidden_edge)
        self.unpick_encoder = gat_encoder.GAT_encoder(node_size=self.input_layer_hidden,
                                                      edge_size=self.input_layer_hidden_edge,
                                                      hidden_size=self.encoder_hidden,
                                                      num_layers=params['gat_layers'],
                                                      nheads=params['gat_nhead'],
                                                      is_mix_attention=params['gat_mix_attn'],
                                                      is_update_edge=params['gat_update_e'],
                                                      num_node=self.max_seq_len)
        if self.hierarchical:
            self.aoi_represent_size = self.encoder_hidden_aoi
            self.input_layer_aoi_edge = nn.Linear(aoi_edge_size, self.input_layer_hidden_aoi_edge)
            self.aoi_encoder = gat_encoder.GAT_encoder(node_size=self.input_layer_hidden_aoi,
                                                       edge_size=self.input_layer_hidden_aoi_edge,
                                                       hidden_size=self.encoder_hidden_aoi,
                                                       num_layers=params['gat_layers_aoi'],
                                                       nheads=params['gat_nhead_aoi'],
                                                       is_mix_attention=params['gat_mix_attn_aoi'],
                                                       is_update_edge=params['gat_update_e_aoi'],
                                                       num_node=self.max_aoi_len)

        self.eta_decoder_input_size = self.package_represent_size + self.pe_size
        if 'eta' in self.eta_aoi_fea_use:
            self.eta_decoder_input_size += 1
        if 'order' in self.eta_aoi_fea_use:
            self.eta_decoder_input_size += self.pe_size
        self.eta_decoder = sortlstm_decoder.lstm_eta_decoder(state_size=self.eta_decoder_input_size,
                                                             hidden_size=self.eta_decoder_hidden,
                                                             seq_len=self.max_seq_len)
        if self.hierarchical:
            self.eta_decoder_input_size_aoi = self.aoi_represent_size + self.pe_size
            self.eta_decoder_aoi = sortlstm_decoder.lstm_eta_decoder(state_size=self.eta_decoder_input_size_aoi,
                                                                     hidden_size=self.eta_decoder_hidden_aoi,
                                                                     seq_len=self.max_aoi_len)

        self.order_decoder_input_size = self.package_represent_size
        if 'eta' in self.order_aoi_fea_use:
            self.order_decoder_input_size += 1
        if 'order' in self.order_aoi_fea_use:
            self.order_decoder_input_size += self.pe_size
        self.order_decoder = PN_decoder(self.order_decoder_input_size,
                                        self.order_decoder_input_size,
                                        courier_dim=courier_size,
                                        max_seq_len=self.max_seq_len)
        if self.hierarchical:
            self.order_decoder_aoi = PN_decoder(self.aoi_represent_size,
                                                self.aoi_represent_size,
                                                courier_dim=courier_size,
                                                max_seq_len=self.max_aoi_len)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def get_model_name(self, params):
        file_name = f"multi_{params['encoder']}_{params['order_decoder']}_{params['eta_decoder']}"
        if self.hierarchical:
            file_name += '_hierarchical'
        return file_name

    def positional_encoding(self, d_model):
        pe_table = []
        for pos in range(self.max_seq_len):
            pos_en = []
            for ii in range(0, int(d_model), 2):
                pos_en.append(math.sin(pos / 10000 ** (2 * ii / d_model)))
                pos_en.append(math.cos(pos / 10000 ** (2 * ii / d_model)))
            pe_table.append(pos_en)
        return torch.FloatTensor(pe_table)

    def forward(self, unpick_fea, edge_fea, unpick_len, courier_fea, global_fea, idx, pos,
                aoi_index=None, aoi_fea=None, aoi_edge=None,
                aoi_len=None, aoi_idx=None, aoi_pos=None, is_train=False, eta=None, aoi_eta=None):
        global_fea = global_fea.unsqueeze(1)

        global_fea_aoi = global_fea.repeat(1, self.max_aoi_len, 1)
        aoi_x = torch.cat([aoi_fea, global_fea_aoi], dim=2)
        aoi_x_input = self.input_layer_aoi(aoi_x)

        # aoi encoder
        aoi_edge_input = self.input_layer_aoi_edge(aoi_edge)
        aoi_represent, aoi_edge = self.aoi_encoder(aoi_x_input, aoi_edge_input)

        # aoi order decoder
        aoi_order_score, aoi_order_arg = self.order_decoder_aoi(aoi_represent, courier_fea, aoi_len)
        aoi_pred_pos = torch.argsort(aoi_order_arg.detach(), dim=1)    # pos
        aoi_sort_info = self.pe_tabel[aoi_pred_pos.long()].float()     # pe
        aoi_eta_input = torch.cat([aoi_represent, aoi_sort_info], dim=2)

        # aoi eta decoder
        aoi_pred_eta = self.eta_decoder_aoi(aoi_eta_input, aoi_len, aoi_order_arg)

        global_fea_order = global_fea.repeat(1, self.max_seq_len, 1)
        unpick_x = torch.cat([unpick_fea, global_fea_order], dim=2)
        unpick_x_input = self.input_layer_unpick(unpick_x)

        # package encoder
        edge_x_input = self.input_layer_edge(edge_fea)
        package_represent, package_edge = self.unpick_encoder(unpick_x_input, edge_x_input)

        # package order decoder
        order_decoder_input = package_represent

        b, seq = aoi_index.shape
        match_aoi_eta = aoi_pred_eta.gather(dim=1, index=aoi_index.long()).detach()
        aoi_index_order = aoi_index.unsqueeze(-1).expand(b, seq, aoi_sort_info.shape[-1]).long()
        match_aoi_order = aoi_sort_info.gather(dim=1, index=aoi_index_order)
        order_decoder_input = torch.cat([order_decoder_input, match_aoi_order], dim=2)
        order_decoder_input = torch.cat([order_decoder_input, match_aoi_eta.unsqueeze(-1)], dim=2)

        # package order decoder
        order_score, order_arg = self.order_decoder(order_decoder_input, courier_fea, unpick_len)
        pred_pos = torch.argsort(order_arg.detach(), dim=1)          # pos
        sort_info = self.pe_tabel[pred_pos.long()].float()           # pe

        eta_decoder_input = package_represent

        eta_decoder_input = torch.cat([eta_decoder_input, match_aoi_order], dim=2)
        eta_decoder_input = torch.cat([eta_decoder_input, match_aoi_eta.unsqueeze(-1)], dim=2)

        eta_decoder_input = torch.cat([eta_decoder_input, sort_info], dim=2)
        pred_eta = self.eta_decoder(eta_decoder_input, unpick_len, order_arg)

        results = {}
        results['eta'] = pred_eta
        results['order'] = order_score
        results['eta_aoi'] = aoi_pred_eta
        results['order_aoi'] = aoi_order_score

        return results
