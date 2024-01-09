import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQNBase2(BaseNetwork):

    def __init__(self, input_dim, num_hidden_units=128, has_speed=True, spd_net_type='cnn'):
        super(DQNBase2, self).__init__()
        self.has_speed = has_speed
        self.spd_net_type = spd_net_type
        self.net = nn.Sequential(
            nn.Linear(input_dim, num_hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.ReLU(inplace=True),
        ).apply(initialize_weights_he)

        if spd_net_type == 'cnn':
            self.speed_sequence_net = nn.Sequential(
                # Adjust kernel size as needed
                nn.Conv1d(1, 10, kernel_size=2),
                nn.ReLU(inplace=True),
                # Adjust kernel size as needed
                nn.Conv1d(10, 10, kernel_size=2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(10)
            ).apply(initialize_weights_he)
        elif spd_net_type == 'lstm':
            self.speed_sequence_net = nn.LSTM(
                input_size=2,      # 每个时间点的特征数量为2（车速，距离）
                hidden_size=64,    # 隐藏单元数量，可以根据需要调整
                num_layers=1,      # LSTM层的数量
                batch_first=True   # 输入和输出的张量以批大小为第一维度
            ).apply(initialize_weights_he)
        if self.spd_net_type == 'lstm':
            self.final_net = nn.Sequential(
                # Combine both inputs
                nn.Linear(num_hidden_units+64, num_hidden_units),
                nn.ReLU(inplace=True),
            ).apply(initialize_weights_he)
        # if self.has_speed:
        #     self.final_net = nn.Sequential(
        #         nn.Linear(num_hidden_units+80, num_hidden_units),  # Combine both inputs
        #         nn.ReLU(inplace=True),
        #     ).apply(initialize_weights_he)
        if not self.has_speed:
            self.final_net = nn.Sequential(
                # Combine both inputs
                nn.Linear(num_hidden_units, num_hidden_units),
                nn.ReLU(inplace=True),
            ).apply(initialize_weights_he)

    def forward(self, states, speed_sequence):
        original_output = self.net(states)
        if self.spd_net_type == 'lstm':
            speed_sequence = speed_sequence.view(speed_sequence.size(
                0), speed_sequence.size(2), 2)  # Add channel dimension
            speed_sequence_output = self.speed_sequence_net(speed_sequence)
            speed_sequence_output = speed_sequence_output[0][:, -1, :]
        elif self.spd_net_type == 'cnn':
            speed_sequence = speed_sequence.view(
                speed_sequence.size(0), 1, -1)  # Add channel dimension
            speed_sequence_output = self.speed_sequence_net(speed_sequence)
            speed_sequence_output = speed_sequence_output.view(
                speed_sequence_output.size(0), -1)
        if self.has_speed:
            combined_output = torch.cat(
                (original_output, speed_sequence_output), dim=1)
        else:
            combined_output = original_output
        return self.final_net(combined_output)


class QNetwork2(BaseNetwork):

    def __init__(self, input_dim, num_actions, shared=False,
                 dueling_net=False, has_speed=True, spd_net_type='cnn'):
        super().__init__()

        if not shared:
            self.conv = DQNBase2(
                input_dim, has_speed=has_speed, spd_net_type=spd_net_type)

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_actions)).apply(initialize_weights_he)
        else:
            self.a_head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_actions)).apply(initialize_weights_he)
            self.v_head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1)).apply(initialize_weights_he)

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states, speed_seq):
        if not self.shared:
            states = self.conv(states, speed_seq)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class CateoricalPolicy2(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False, spd_type='cnn', has_speed=True):
        super().__init__()
        if not shared:
            self.conv = DQNBase2(
                num_channels, has_speed=has_speed, spd_net_type=spd_type)

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions))

        self.shared = shared

    def act(self, states, speed_seq):
        if not self.shared:
            states = self.conv(states, speed_seq)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states, speed_seq):
        if not self.shared:
            states = self.conv(states, speed_seq)

        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class TwinnedQNetwork2(BaseNetwork):
    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False, has_speed=True, spd_type='cnn'):
        super().__init__()
        self.Q1 = QNetwork2(num_channels, num_actions, shared,
                            dueling_net, has_speed=has_speed, spd_net_type=spd_type)
        self.Q2 = QNetwork2(num_channels, num_actions, shared,
                            dueling_net, has_speed=has_speed, spd_net_type=spd_type)

    def forward(self, states, speed_seq):
        q1 = self.Q1(states, speed_seq)
        q2 = self.Q2(states, speed_seq)
        return q1, q2


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
