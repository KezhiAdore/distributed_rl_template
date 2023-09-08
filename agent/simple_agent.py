from .base_agent import BaseAgent
import random

class SimpleAgent(BaseAgent):
    def __init__(self, name, device, buffer_size) -> None:
        super().__init__(name, device, buffer_size)
        
    def act(self, obs):
        return random.randint(0, 1)
    
    def update(self, batch):
        print(f"agent update, batch size:{len(batch)}")
        return random.random()
    
    def get_weights(self):
        print("获取agent参数")
        return None
    
    def set_weights(self, weights):
        print("设置agent参数")
        return None