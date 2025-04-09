from abc import ABC,abstractmethod

class Agent_base(ABC):
    def __init__(self,cfg):
        self.cfg=cfg

    @abstractmethod
    def choose_action(self,state, c=None):
        pass

    @abstractmethod
    def learn(self,global_step=None):
        pass

    #
    # @abstractmethod
    # def save_agent():
    #     return

    #
    # @abstractmethod
    # def load_agent():
    #     return

    #@abstractmethod
    #def update_target_networks(self):
    #    return
