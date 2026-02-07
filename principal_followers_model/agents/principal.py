import agentpy as ap
import random
from collections import Counter


class Principal(ap.Agent):
    def setup(self):
        self.last_plan = None

    def propose_allocation(self):
        """
        Returns a dict follower_id -> place_index (0..L-1)
        Basic policy: assign agents to fill capacities as well as possible.
        This can be replaced with learning/optimization later.
        """
        L = self.model.p.L
        caps = list(self.model.p.capacities)

        # Followers to assign (exclude principal itself)
        follower_ids = [a.id for a in self.model.followers]

        # Strategy: build a "target list" that repeats each place by its capacity,
        # then fill remaining with random places.
        target_places = []
        for j in range(L):
            target_places.extend([j] * max(0, int(caps[j])))

        # If total capacity < number of followers, fill remainder randomly
        while len(target_places) < len(follower_ids):
            target_places.append(random.randrange(L))

        # If total capacity > number of followers, truncate
        target_places = target_places[:len(follower_ids)]

        random.shuffle(target_places)
        plan = {fid: target_places[i] for i, fid in enumerate(follower_ids)}
        self.last_plan = plan
        return plan
