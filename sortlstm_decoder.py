import torch
import torch.nn as nn
import torch.nn.functional as F


class lstm_eta_decoder(nn.Module):
    def __init__(self, state_size, hidden_size, seq_len=20):
        super(lstm_eta_decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=state_size,
                            hidden_size=hidden_size,
                            num_layers=2, batch_first=True)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.seq_len = seq_len

    def forward(self, hidden_state, unpick_len, pred_idx, pred_score=None):
        pred_idx[pred_idx == -1] = self.seq_len - 1
        state_sort_idx = pred_idx.unsqueeze(-1).expand(pred_idx.shape[0], pred_idx.shape[1], hidden_state.shape[-1])
        sorted_state = hidden_state.gather(1, state_sort_idx.to(torch.int64))

        pack_state = nn.utils.rnn.pack_padded_sequence(sorted_state,
                                                       unpick_len.cpu().to(torch.int64),
                                                       batch_first=True,
                                                       enforce_sorted=False)
        output_state, (_, _) = self.lstm(pack_state)
        output_state, _ = nn.utils.rnn.pad_packed_sequence(output_state, batch_first=True)
        output_state = nn.functional.pad(output_state,
                                         [0, 0, 0, self.seq_len - output_state.shape[1], 0, 0],
                                         mode="constant",
                                         value=0)

        pred_eta = self.output_layer(output_state)
        pred_eta = pred_eta.squeeze()

        resort_index = torch.argsort(pred_idx, dim=1)
        resorted_pred_eta = pred_eta.gather(1, resort_index.to(torch.int64))
        return resorted_pred_eta
