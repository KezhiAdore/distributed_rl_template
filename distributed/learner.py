import ray
from torch.utils.tensorboard import SummaryWriter

from .communication_server import CommunicationServer
from agent import BaseAgent

@ray.remote
class Learner:
    def __init__(self,
                 agent: BaseAgent,
                 com_server: CommunicationServer,
                 update_num: int,   # 每次更新的更新次数
                 model_dir: str,
                 cur_time,
                 is_clear_buffer: bool=False # 是否在每次更新后清空经验池
                 ) -> None:
        self._agent = agent
        self._com_server = com_server
        self._update_num = update_num
        self._model_dir = model_dir
        self._is_clear_buffer = is_clear_buffer
        
        self.writer = SummaryWriter(f'./log/{cur_time.day}_{cur_time.hour}_{cur_time.minute}/learner')
        self._num_update = 0
        
        # 初始化时同步一次参数
        self._com_server.set_weights.remote(self._agent.get_weights())
    
    def update(self, batch_size):
        loss = 0
        for _ in range(self._update_num):
            # 从经验池中采样数据
            batch = ray.get(self._com_server.sample.remote(batch_size))
            # 更新网络
            loss += self._agent.update(batch)
        loss/=self._update_num
        
        # 同步权重
        self._com_server.set_weights.remote(self._agent.get_weights())
        
        # 根据配置决定是否清空经验池
        if self._is_clear_buffer:
            self._com_server.clear_buffer.remote()
        
        self.writer.add_scalar('loss', loss, self._num_update)
        
        self._num_update += 1
    
    def save(self, prefix):
        pass
    
    def load(self, prefix, episode):
        pass
    