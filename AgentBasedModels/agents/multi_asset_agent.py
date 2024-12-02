import agentpy as ap

# input de estrategia es la memoria de un asset i
# output prediccion de la asistencia en el tiempo t + 1, num entre 0 y el numero de agentes seguidores

class MultiAssetAgent(ap.Agent):
    def setup(self):
        self.u = [0] * self.p.M
        self.strategies = self.p.strategies

    def rank_strategies(self, memory):
        # returna la utilidad acumulada de cada estrategia y ordena de mayor a menor
        # [(10, 'avg_assistance'), (5, 'random_assistance'), (3, 'last_time_assistance')]
        # (1-2 * uij)*(ajt - lj)
        # 4 casos
        # 1. uij = 0 (positivo) y no se satura (negativo) => (negativo)
        # 2. uij = 1 (negativo) y no se satura (negativo) => (positivo)
        # 3. uij = 0 (positivo) y se satura  (positivo) => (positivo)
        # 4. uij = 1 (negativo) y se satura (positivo) => (negativo)

        self.strategies

    def choose_strategy(self, memory):
        max(self.rank_strategies(memory))

    def prediction(self, asset_id, asset_memory, strategy):
        self.u[asset_id] = strategy.choose_action(asset_memory)

    # iterar sobre cada asset
    #   actualizar el rankeo
    #   elegir la estrategia con mayor rankeo
    #    correr la prediccion con la estrategia elegida
    #    comparar la prediccion vs el limite del asset (?)
    #    realizar accion (ie. devolver 0 o 1)
    def action(self, assets_memories):
        for asset_id, asset_memory in enumerate(assets_memories):
            self.update_ranking(asset_memory)
            strategy = self.choose_strategy()
            self.prediction(asset_id, asset_memory, strategy)

