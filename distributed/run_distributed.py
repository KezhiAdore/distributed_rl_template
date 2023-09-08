from distributed import Worker, Learner, CommunicationServer
import datetime
import ray

def run_distributed_train(worker_num,
                          env_class, 
                          env_args, 
                          agent_class, 
                          agent_args,
                          total_episode: int,   # 需要收集经验的总回合数
                          model_dir: str,
                          buffer_size: int,  # 经验池大小
                          batch_size: int,  # 每次更新采样的经验数量
                          min_train_buffer_len:int,   # 最小训练缓存数量
                          update_num: int=1,
                          is_clear_buffer: bool=False,
                          ):
    cur_time = datetime.datetime.now()
    # 创建worker, learner, communication_server
    print(f"----------初始化进程----------")
    com_server = CommunicationServer.remote(worker_num, cur_time, buffer_size)
    workers = [Worker.remote(id, env_class, env_args, agent_class, agent_args, com_server, cur_time) for id in range(worker_num)]
    agent = agent_class(**agent_args)
    learner = Learner.remote(agent, com_server, update_num, model_dir, cur_time, is_clear_buffer)
    
    print(f"----------Worker收集经验----------")
    for worker in workers:
        worker.collect_trajectories.remote(total_episode)
        
    print(f"----------Learner开始训练----------")
    while True:
        buffer_len = ray.get(com_server.buffer_len.remote())
        total_episode_now = ray.get(com_server.get_total_episode.remote())
        if (buffer_len > min_train_buffer_len) and (total_episode_now % worker_num < 2):
            learner.update.remote(batch_size)
        
        # 保存模型
        pass
    
        # 结束训练
        if total_episode_now >= total_episode:
            break
    