import torch
import torch.nn as nn
import torch.nn.functional as F

from projections import COMP_DIM


STATE_RAW_DIM = COMP_DIM * 4 + 1 + 3  # v_s + txt_s + a_s + h_prev + t/T + 3 norms = 1028
STATE_DIM = 512
LSTM_INPUT_DIM = COMP_DIM * 3        # 768
LSTM_HIDDEN_DIM = COMP_DIM           # 256


class StateEncoder(nn.Module):
    def __init__(self, in_dim=STATE_RAW_DIM, out_dim=STATE_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_uniform_(self.proj.weight, a=0, nonlinearity='linear')
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return F.relu(self.proj(x))


class Actor(nn.Module):
    """Outputs Dirichlet concentrations over 3 modalities.

    Final layer uses small-std init so initial alphas ≈ 1 (uniform prior)."""

    def __init__(self, in_dim=STATE_DIM, hidden=256, num_modalities=3, final_std=0.01):
        # başlangıçta tüm modelitelerin eşit ağırlıklı olması için alpha=1 olmalıdır. Bunu için final_std çok küçük tutulur.
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, num_modalities)
        nn.init.normal_(self.head.weight, std=final_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, state):
        h = self.net(state)
        raw = self.head(h)
        alpha = F.softplus(raw) + 1
        # alpha'yı kafese al: uçlara kaçarsa Dirichlet entropisi patlar (loss'u bozar).
        alpha = alpha.clamp(max=20.0)
        return alpha


class Critic(nn.Module):
    def __init__(self, in_dim=STATE_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class RLAgent(nn.Module):
    """Bundles state encoder, LSTM, actor, critic for joint optimisation."""

    def __init__(self, lstm_input_dim=LSTM_INPUT_DIM, lstm_hidden_dim=LSTM_HIDDEN_DIM,
                 state_raw_dim=STATE_RAW_DIM, state_dim=STATE_DIM, num_modalities=3):
        super().__init__()
        self.lstm = nn.LSTMCell(lstm_input_dim, lstm_hidden_dim)
        self.state_encoder = StateEncoder(state_raw_dim, state_dim)
        self.actor = Actor(state_dim, num_modalities=num_modalities)
        self.critic = Critic(state_dim)
        self.lstm_hidden_dim = lstm_hidden_dim

    def init_state(self, device):
        # LSTM in her yeni video için hidden state ve cell state sıfırlanır
        h0 = torch.zeros(1, self.lstm_hidden_dim, device=device)
        c0 = torch.zeros(1, self.lstm_hidden_dim, device=device)
        return h0, c0

    def step(self, v_small, txt_small, a_small, h_prev, c_prev, t_norm, norms):
        """Single-frame forward.

        Args:
            v_small, txt_small, a_small: (1, COMP_DIM) compressed features.
            h_prev, c_prev: (1, LSTM_HIDDEN_DIM) recurrent state from previous frame.
            t_norm: scalar tensor () — t/T.
            norms: (3,) tensor of ||v||, ||txt||, ||a||.

        Returns:
            alpha (1, 3), value (1,), (h_new, c_new).
        """
        lstm_in = torch.cat([v_small, txt_small, a_small], dim=-1)
        h_new, c_new = self.lstm(lstm_in, (h_prev, c_prev))
        state_raw = torch.cat([
            v_small, txt_small, a_small, h_prev,
            t_norm.view(1, 1), norms.view(1, 3)
        ], dim=-1)
        state = self.state_encoder(state_raw)
        alpha = self.actor(state)
        value = self.critic(state)
        return alpha, value, h_new, c_new, state
