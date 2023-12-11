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


class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)

class DQNBase2(BaseNetwork):
    
    def __init__(self, input_dim,num_hidden_units=128, has_speed = True):
        super(DQNBase2, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, num_hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.ReLU(inplace=True),
        ).apply(initialize_weights_he)

        self.speed_sequence_net = nn.Sequential(
            nn.Conv1d(1,10, kernel_size=10),  # Adjust kernel size as needed
            nn.ReLU(inplace=True),
            nn.Conv1d(10,10, kernel_size=10),  # Adjust kernel size as needed
            nn.ReLU(inplace=True),
            nn.MaxPool1d(10)
        ).apply(initialize_weights_he)

        self.final_net = nn.Sequential(
            nn.Linear(num_hidden_units+80, num_hidden_units),  # Combine both inputs
            nn.ReLU(inplace=True),
        ).apply(initialize_weights_he)

    def forward(self, states, speed_sequence):
        original_output = self.net(states)
        speed_sequence = speed_sequence.view(speed_sequence.size(0), 1, -1)  # Add channel dimension
        speed_sequence_output = self.speed_sequence_net(speed_sequence)
        speed_sequence_output = speed_sequence_output.view(speed_sequence_output.size(0), -1)
        if self.has_speed:
            combined_output = torch.cat((original_output, speed_sequence_output), dim=1)
        else:
            combined_output = original_output
        return self.final_net(combined_output)

class QNetwork2(BaseNetwork):
    
    def __init__(self, input_dim, num_actions, shared=False,
                 dueling_net=False,has_speed = True):
        super().__init__()

        if not shared:
            self.conv = DQNBase2(input_dim,has_speed=has_speed)

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

    def forward(self, states,speed_seq):
        if not self.shared:
            states = self.conv(states,speed_seq)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)

class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        if not shared:
            self.conv = DQNBase(num_channels)

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_channels, num_actions, shared, dueling_net)
        self.Q2 = QNetwork(num_channels, num_actions, shared, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False):
        super().__init__()
        if not shared:
            self.conv = DQNBase(num_channels)

        self.head = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions))

        self.shared = shared

    def act(self, states):
        if not self.shared:
            states = self.conv(states)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        if not self.shared:
            states = self.conv(states)

        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

class CateoricalPolicy2(BaseNetwork):
    
    def __init__(self, num_channels, num_actions, shared=False):
        super().__init__()
        if not shared:
            self.conv = DQNBase2(num_channels)

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions))

        self.shared = shared

    def act(self, states,speed_seq):
        if not self.shared:
            states = self.conv(states,speed_seq)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states,speed_seq):
        if not self.shared:
            states = self.conv(states,speed_seq)

        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
    
class TwinnedQNetwork2(BaseNetwork):
    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork2(num_channels, num_actions, shared, dueling_net)
        self.Q2 = QNetwork2(num_channels, num_actions, shared, dueling_net)

    def forward(self, states,speed_seq):
        q1 = self.Q1(states,speed_seq)
        q2 = self.Q2(states,speed_seq)
        return q1, q2

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
