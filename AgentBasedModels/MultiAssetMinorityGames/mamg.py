import agentpy as ap

from agents.multi_asset_agent import MultiAssetAgentPrincipal, MultiAssetAgentFollower
from memories.circular_buffer_memory import CircularBufferMemory
from strategies.random_strategy import RandomStrategy

class MultiAssetMinorityGame(ap.Model):
    def setup(self):
        # Add support for multiple type of agents
        # self.agents = ap.AgentList(self, self.p.agents, MultiAssetAgentPrincipal, MultiAssetAgentFollower)

        self.follower_agents = ap.AgentList(self, self.p.agents, MultiAssetAgentFollower)
        self.principal = MultiAssetAgentPrincipal(self)

        self.agents = self.follower_agents + self.principal
        # We have M memories of size m
        self.assets_memories = [CircularBufferMemory(self.p.m) for asset in range(self.p.M)]
        self.strategies = [self.get_strategy(strategy) for strategy in self.p.strategies]

    def step(self):
        """ Call a method for every agent. """
        self.agents.action(self.assets_memories)

    def update(self):
        """ Record a dynamic variable. """
        self.agents.record('utility')

    def end(self):
        """ Repord an evaluation measure. """
        self.report('acc_utility', 1)

    def get_strategy(self, strategy):
        if strategy == 'random':
            return (RandomStrategy())

        