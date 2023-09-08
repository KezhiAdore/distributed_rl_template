from tianshou.data import ReplayBuffer, Batch

class BaseAgent:
    def __init__(self,
                 name,
                 device,
                 buffer_size,
                 ) -> None:
        self._name = name
        self._device = device
        self._replay_buffer = ReplayBuffer(buffer_size)
    
    def reset(self):
        self._replay_buffer.reset()
    
    def act(self, obs):
        raise NotImplementedError("step method not implemented")
    
    def update(self, batch):
        raise NotImplementedError("update method not implemented")
    
    def get_weights(self):
        raise NotImplementedError("get_weights method not implemented")
    
    def set_weights(self, weights):
        raise NotImplementedError("set_weights method not implemented")
    
    def store(self, obs, act, rew, obs_next, done):
        batch = Batch({
            "obs": obs,
            "act": act,
            "rew": rew,
            "terminated": done,
            "obs_next": obs_next,
            "truncated": False,
        })
        self._replay_buffer.add(batch)
    
    def get_buffer(self):
        return self._replay_buffer
    