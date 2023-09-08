import ray
import gym
from distributed import run_distributed_train
from agent import SimpleAgent


if __name__ == "__main__":
    # 初始化Ray
    # ray.init(local_mode=True)   # 调试模式
    ray.init()
    
    env_args = {
        "id": "CartPole-v1"
    }
    agent_args = {
        "name": "simple",
        "device": "cpu",
        "buffer_size": 10000
    }
    run_distributed_train(
        worker_num=8,   # worker数量
        env_class=gym.make, # 环境类
        env_args=env_args,  # 环境初始化参数
        agent_class=SimpleAgent,    # 智能体类
        agent_args=agent_args,  # 智能体初始化参数
        total_episode=100000,   # worker总共收集多少个episode的经验
        model_dir="./model",    # 模型保存路径
        buffer_size=100000, # 经验池大小
        batch_size=1024,    # 每次更新的batch大小
        min_train_buffer_len=10000, # 经验池中的经验数大于该值时才开始更新
        update_num=10,  # 每次Learner更新的更新次数
        is_clear_buffer=False,  # 每次Learner更新后是否清空经验池
    )
    