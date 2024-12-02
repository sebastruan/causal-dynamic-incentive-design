
from MultiAssetMinorityGames.mamg import MultiAssetMinorityGame


parameters = {
    'N': 100, # n agents
    'M': 3, # m assets
    'm': 10, # memory size,
    'max_assets_capacities': [20, 30, 25], # this can be randomized, the sum <= N
    'strategies': ['avg_assintance', 'random_assistance', 'last_time_assistance']
}

model = MultiAssetMinorityGame(parameters)
results = model.run()
