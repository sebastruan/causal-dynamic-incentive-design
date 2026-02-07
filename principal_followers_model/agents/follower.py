import agentpy as ap
import random


class Follower(ap.Agent):
    def setup(self):
        self.place = None
        self.reward = 0.0
        self.total_reward = 0.0

    def decide_place(self, suggested_place: int):
        """
        With probability influence_strength, follow the principal.
        Otherwise choose according to a fallback policy.
        """
        L = self.model.p.L
        infl = self.model.p.influence_strength
        fallback = self.model.p.follower_fallback  # "random" or "least_crowded"

        if random.random() < infl:
            return suggested_place

        # Deviate:
        if fallback == "least_crowded":
            # choose a least-crowded place based on current occupancy
            occ = self.model.place_occupancy()
            min_occ = min(occ.values())
            candidates = [j for j, c in occ.items() if c == min_occ]
            return random.choice(candidates)
        else:
            return random.randrange(L)
