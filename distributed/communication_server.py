import ray
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import VectorReplayBuffer, Batch

@ray.remote
class CommunicationServer:
    """用于worker间的通信以及woker与learner之间的参数同步
    """
    
    def __init__(self, worker_num, cur_time, buffer_size: int=100000) -> None:
        # 初始化需要通信的变量和参数
        self._total_episode = 0
        self._worker_num = worker_num
        self._weights = None
        self._replay_buffer = VectorReplayBuffer(buffer_size, self._worker_num)
        
        self.writer = SummaryWriter(f'./log/{cur_time.day}_{cur_time.hour}_{cur_time.minute}')
    
    def total_episode_add1(self):
        self._total_episode += 1
        return self._total_episode
    
    def get_total_episode(self):
        return self._total_episode
    
    def set_weights(self, weights):
        self._weights = weights
        
    def get_weights(self):
        return self._weights
    
    def store(self, batch:Batch, worker_id):
        self._replay_buffer.add(batch, [worker_id] * len(batch))
    
    def buffer_len(self):
        return self._replay_buffer.__len__()
    
    def clear_buffer(self):
        self._replay_buffer.reset()
    
    def sample(self, batch_size):
        return self._replay_buffer.sample(batch_size)
    
    def print_tb(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    