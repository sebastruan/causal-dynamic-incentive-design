import agentpy as ap
from model.principal_followers_model import PrincipalFollowersModel  # if you saved it as a module

parameters = {
    'n': 10,
    'L': 4,
    'capacities': (3, 2, 2, 3),
    'steps': 5,

    'influence_strength': ap.Values(0.2, 0.5, 0.8, 0.95),
    'follower_fallback': ap.Values('random', 'least_crowded'),

    'r_match': 10.0,
    'r_under': 1.0,
    'r_over': -5.0,
}

exp = ap.Experiment(PrincipalFollowersModel, parameters, iterations=20, randomize=True)
results = exp.run()
