import ray
from torch.utils.tensorboard import SummaryWriter
from .communication_server import CommunicationServer

@ray.remote
class Worker:
    def __init__(self, 
                 id, 
                 env_class, # 环境类
                 env_args: dict,  # 环境初始化参数
                 agent_class,   # 智能体类
                 agent_args: dict,    # 智能体初始化参数
                 com_server:CommunicationServer, 
                 cur_time,
                 ) -> None:
        self._id = id
        self._env_class = env_class
        self._env_args = env_args
        self._agent_class = agent_class
        self._agent_args = agent_args
        self._com_server = com_server
        
        self._num_episode = 0   # 用于统计当前worker已经收集了多少个episode的经验
        self._total_episode = 0 # 获取当前所有worker收集了多少个episode的经验
        
        self.writer = SummaryWriter(f'./log/{cur_time.day}_{cur_time.hour}_{cur_time.minute}/{self._id}')
    
    def collect_trajectories(self, total_episode):
        while self._total_episode <= total_episode:
            try:
                print('-' * 50)
                print(f'-------------worker{self._id}开启第{self._num_episode}次轨迹------')
                
                # 初始化agent和env
                agent = self._agent_class(**self._agent_args)
                env = self._env_class(**self._env_args)
                
                # 获取最新参数并对worker中的智能体进行更新
                print(f'----worker{self._id}获取learner网络参数----')
                cur_weights = ray.get(self._com_server.get_weights.remote())
                if cur_weights:
                    agent.set_weights(cur_weights)
                
                # 与环境交互收集轨迹
                print(f'----worker{self._id}开始收集轨迹----')
                obs, _ = env.reset()
                step = 0
                total_rew = 0
                done = False
                while step < 100000:
                    # 智能体与环境交互
                    act = agent.act(obs)
                    obs_next, rew, done, _, info = env.step(act)
                    agent.store(obs, act, rew, obs_next, done)
                    obs = obs_next
                    # 计算累计回报，更新步数
                    total_rew += rew
                    step += 1
                    # 环境结束
                    if done:
                        self.writer.add_scalar('total_rew', total_rew, self._num_episode)
                        self.writer.add_scalar('total_step', step, self._num_episode)
                        break
                
                # 将agent buffer存入communication server
                print(f'----worker{self._id}将轨迹存入buffer----')
                buffer = agent.get_buffer()
                batch, _ = buffer.sample(0)
                self._com_server.store.remote(batch, self._id)
                
                # 更新轨迹数
                self._num_episode += 1
                self._total_episode = ray.get(self._com_server.total_episode_add1.remote())
                buffer_len = ray.get(self._com_server.buffer_len.remote())
                
                print(f'worker{self._id}第{self._num_episode}条轨迹收集完成,该轨迹长度为{step}, \
                    总收集轨迹数量为{self._total_episode},'f'总buffer长度为{buffer_len}')
            except Exception:
                print(f'worker{self._id}第{self._num_episode}条轨迹收集失败，重启收集')
                continue
        
        print(f'Worker线程{self._id}结束,共执行{self._num_episode}轨迹，关闭环境')
        
        
        