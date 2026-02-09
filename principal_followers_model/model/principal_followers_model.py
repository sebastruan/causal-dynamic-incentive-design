from collections import Counter
import random
import agentpy as ap
from agents.principal import Principal
from agents.follower import Follower
from lib.rewards import place_reward

class PrincipalFollowersModel(ap.Model):

    def setup(self):
        # Create agents
        self.principal = Principal(self)
        self.followers = ap.AgentList(self, self.p.n - 1, Follower)

        # Places are just indexed 0..L-1 with capacities
        assert len(self.p.capacities) == self.p.L, "capacities must have length L"
        
        print(f"Model setup: {self.p.n} agents (1 principal + {self.p.n - 1} followers), "
              f"{self.p.L} places with capacities {self.p.capacities}, "
              f"influence strength {self.p.influence_strength}, "
              f"follower fallback '{self.p.follower_fallback}'")
        # State: where each agent is (place index)
        # We'll keep follower places; principal can also occupy a place if you want.
        for f in self.followers:
            f.place = random.randrange(self.p.L)

        # For logging
        self.total_system_reward = 0.0
        self.history_system_reward = []
        self.history_occupancies = []

    def place_occupancy(self):
        """Return {place_index: count} based on current follower placements."""
        counts = Counter([f.place for f in self.followers])
        return {j: counts.get(j, 0) for j in range(self.p.L)}

    def step(self):
        # 1) Principal proposes an allocation for followers
        plan = self.principal.propose_allocation()

        # 2) Followers choose a place given the suggestion + noise
        # plan maps follower_id -> suggested_place
        for f in self.followers:
            suggested = plan[f.id]
            f.place = f.decide_place(suggested)

        # 3) Compute occupancy per place after movement
        occ = self.place_occupancy()

        # 4) Compute place rewards (shared environment reward)
        place_rewards = {}
        for j in range(self.p.L):
            place_rewards[j] = place_reward(
                occupancy=occ[j],
                capacity=int(self.p.capacities[j]),
                r_match=self.p.r_match,
                r_under=self.p.r_under,
                r_over=self.p.r_over
            )

        # 5) Assign agent rewards (simple: each follower gets reward of the place they chose)
        system_reward = 0.0
        for f in self.followers:
            f.reward = place_rewards[f.place]
            f.total_reward += f.reward
            system_reward += f.reward

        self.total_system_reward += system_reward
        self.history_system_reward.append(system_reward)
        self.history_occupancies.append(occ)

        # Record observations (AgentPy DataDict)
        self.record('system_reward', system_reward)
        for j in range(self.p.L):
            self.record(f'occupancy_place_{j}', occ[j])

    def end(self):
        # Useful summary metrics
        avg_step_reward = self.total_system_reward / max(1, self.p.steps)
        self.record('total_system_reward', self.total_system_reward)
        self.record('avg_step_reward', avg_step_reward)

        # Also record follower totals (mean)
        mean_follower_total = sum(f.total_reward for f in self.followers) / max(1, len(self.followers))
        self.record('mean_follower_total_reward', mean_follower_total)
        print(f"Model ended. Total system reward: {self.total_system_reward}, "
              f"Average step reward: {avg_step_reward}, "
              f"Mean follower total reward: {mean_follower_total}") 


