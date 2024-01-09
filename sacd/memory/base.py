from collections import deque
import numpy as np
import torch
import pickle


class MultiStepBuff:

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.reset()

    def append(self, state, speed_seq, action, reward):
        self.states.append(state)
        self.speed_seqs.append(speed_seq)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self, gamma=0.99):
        assert len(self.rewards) > 0
        state = self.states.popleft()
        speed_seq = self.speed_seqs.popleft()
        action = self.actions.popleft()
        reward = self._nstep_return(gamma)
        return state,speed_seq, action, reward

    def _nstep_return(self, gamma):
        r = np.sum([r * (gamma ** i) for i, r in enumerate(self.rewards)])
        self.rewards.popleft()
        return r

    def reset(self):
        # Buffer to store n-step transitions.
        self.states = deque(maxlen=self.maxlen)
        self.speed_seqs = deque(maxlen=self.maxlen)
        self.actions = deque(maxlen=self.maxlen)
        self.rewards = deque(maxlen=self.maxlen)

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return len(self.rewards) == self.maxlen

    def __len__(self):
        return len(self.rewards)



class LazyMemory(dict):

    def __init__(self, capacity, state_shape, speed_seq_shape, device):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.speed_seq_shape = speed_seq_shape
        self.device = device
        self.reset()

    def reset(self):
        self['state'] = []
        self['speed_seq1'] = []
        self['next_state'] = []
        self['speed_seq2'] = []

        self['action'] = np.empty((self.capacity, 1), dtype=np.int64)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)

        self._n = 0
        self._p = 0

    def append(self, state, speed_seq1, action, reward, next_state,speed_seq2, done,
               episode_done=None):
        self._append(state, speed_seq1, action, reward, next_state, speed_seq2, done)

    def _append(self, state, speed_seq1, action, reward, next_state, speed_seq2, done):
        self['state'].append(state)
        self['speed_seq1'].append(speed_seq1)
        self['next_state'].append(next_state)
        self['speed_seq2'].append(speed_seq2)
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        self.truncate()

    def truncate(self):
        while len(self['state']) > self.capacity:
            del self['state'][0]
            del self['speed_seq1'][0]
            del self['next_state'][0]
            del self['speed_seq2'][0]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty(
            (batch_size, self.state_shape), dtype=float)
        speed_seq1s = np.empty(
            #(batch_size, self.speed_seq_shape), dtype=float)
            (batch_size, self.speed_seq_shape[0],self.speed_seq_shape[1]), dtype=float)
        next_states = np.empty(
            (batch_size, self.state_shape), dtype=float)
        speed_seq2s = np.empty(
            #(batch_size, self.speed_seq_shape), dtype=float)
            (batch_size, self.speed_seq_shape[0],self.speed_seq_shape[1]), dtype=float)

        for i, index in enumerate(indices):
            _index = np.mod(index+bias, self.capacity)
            states[i, ...] = self['state'][_index]
            speed_seq1s[i, ...] = self['speed_seq1'][_index]
            next_states[i, ...] = self['next_state'][_index]
            speed_seq2s[i, ...] = self['speed_seq2'][_index]

        states = torch.tensor(states).to(self.device).float()
        speed_seq1s = torch.tensor(speed_seq1s).to(self.device).float()
        next_states = torch.tensor(next_states).to(self.device).float()
        speed_seq2s = torch.tensor(speed_seq2s).to(self.device).float()
        actions = torch.tensor(self['action'][indices]).to(self.device).float()
        rewards = torch.tensor(self['reward'][indices]).to(self.device).float()
        dones = torch.tensor(self['done'][indices]).to(self.device).float()

        return states,speed_seq1s, actions, rewards, next_states, speed_seq2s, dones

    def __len__(self):
        return self._n


class LazyMultiStepMemory(LazyMemory):

    def __init__(self, capacity, state_shape, speed_seq_shape, device, gamma=0.99,
                 multi_step=3):
        super(LazyMultiStepMemory, self).__init__(
            capacity, state_shape, speed_seq_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)
            
    def save_replay_buffer(self, filename):
            # Serialize the entire memory (including states and sequences)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_replay_buffer(cls, filename):
        # Deserialize the memory from the file
        with open(filename, 'rb') as f:
            replay_buffer = pickle.load(f)
        return replay_buffer

    def append(self, state, speed_seq1, action, reward, next_state, speed_seq2, done):
        if self.multi_step != 1:
            self.buff.append(state, speed_seq1, action, reward)

            if self.buff.is_full():
                state, speed_seq1, action, reward = self.buff.get(self.gamma)
                self._append(state,speed_seq1, action, reward, next_state, speed_seq2, done)

            if done:
                while not self.buff.is_empty():
                    state,speed_seq1, action, reward = self.buff.get(self.gamma)
                    self._append(state,speed_seq1, action, reward, next_state,speed_seq2, done)
        else:
            self._append(state,speed_seq1, action, reward, next_state,speed_seq2, done)
