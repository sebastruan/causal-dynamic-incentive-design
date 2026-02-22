import agentpy as ap

from agents.multi_asset_agent import MultiAssetAgent

class MultiAssetAgentFollower(MultiAssetAgent):
    def setup(self):
        
        self.my_attribute = 0

    def run_prediction(self):
        """ Run a prediction algorithm. """
        self.my_attribute = 1

    def choose_strategy(self):
        """ Choose a strategy. """
        self.my_attribute = 2

    def run_action(self):
        self.run_prediction()
        self.choose_strategy()
