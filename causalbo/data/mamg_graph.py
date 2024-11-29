from pandas import DataFrame
from random import randrange
from causalbo.do_calculus import SCM
import torch

class MAMGGraph(object):
    # epsilon_X

    def U1(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        return input_tensor

    def U2(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        return input_tensor
    def U3(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        return input_tensor
    def U4(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        return input_tensor
    def U5(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        return input_tensor
    def U6(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        return input_tensor
    def U7(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        return input_tensor
    def U8(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        return input_tensor
    def U9(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        return input_tensor
    def U10(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        return input_tensor

    # U accion agente
    # U1 acction agent 1
    # U2 acction agent 2

    # JA utilidad agente
    # JA1 utilidad agente 1
    # JA2 utilidad agente 2

    # ja_1 = (1-2*Uij) * (sum(Uj) - Lj)
    # ja_2 = sum((1-2*Uij) * (sum(Uj) - Lj)) for cada asset

    # input:
    # num_agents = 10
    # num_assets = 3
    # limites_assets = [L1 = 2, L2 = 5, L3 = 4]
    # obs1 = [4,5,6,3,2,1,2,4,5,7]

    # 4 100
    # 5 101
    # 6 110
    # 3 011
    # 2 010
    # 1 001
    # 2 010
    # 4 100
    # 5 101
    # 7 111
    
    
    # output:
    # sum(Uj) - Lj) for all agents
    # attendace_assets = [A1 = 5, A2 = 4, A3 = 6]
    # attendace_assets - limits = [A1 - L1 = 5 - 2 = 3, A2  - L2 = 4 - 5 = -1, A3 - L3 = 6 - 2 = 4] 

    # (1-2*Uij) for each agent
    # obs 1 agent 1 = 4 100
    # U11 = 0
    # U12 = 0
    # U13 = 1
    # (1-2  * U11) = (1-2*0) = 1
    # (1-2  * U12) = (1-2*0) = 1
    # (1-2  * U13) = (1-2*1) = -1

    # (1-2*Uij) * (sum(Uj) - Lj) for each agent
    # obs 1 agent 1 = 4 100
    # (1-2  * U11) * (A1 - L1) = 1 * 3 = 3
    # (1-2  * U12) * (A2 - L2) = 1 * -1 = -1
    # (1-2  * U13) * (A3 - L3) = -1 * 4 = -4

    # JA1 = sum((1-2*Uij) * (sum(Uj) - Lj)) for each agent utilidad_agente_1
    # obs 1 agent 1 = 4 100
    # utilidad_agente_1 = 3 -1 -4 = -2

    #[utilidad_agente_1, utilidad_agente_2, utilidad_agente_3, ...]

    # utilidades_obs1 = [utilidad_agente_1, utilidad_agente_2, utilidad_agente_3, ...]


    # output final: [utlidades_obs1, utilidades_obs2, utilidades_obs3, ..., utilidades_obs100]
    def JA(input_tensor, noise_mean=0, noise_stdev=0):
        assets = 3
        ja_list = []
        all_utilidades = []
        for obs_idx in range(len(input_tensor)):
            utilidades_obs = []
            obs = input_tensor[obs_idx] #[4,5]
            # ja_per_agent = [ja_agent1, ja_agent_2,...]
            ja_per_agent = []
            for agent_index in range(len(obs)):
                agent_decisions = obs[agent_index] #[4]
                agent_decisions_decoded = format(int(agent_decisions), '03b')
                ja = 0
                for asset_idx in range(assets):
                    if asset_idx == 0 and agent_decisions_decoded[asset_idx] == '1': 
                        ja += (1 - 2 * int(agent_decisions_decoded[asset_idx])) # multiply by sum(Uj) - Lj
                    elif asset_idx == 1 and agent_decisions_decoded[asset_idx] == '1':
                        ja += (1 - 2 * int(agent_decisions_decoded[asset_idx])) # multiply by sum(Uj) - Lj
                    elif asset_idx == 2 and agent_decisions_decoded[asset_idx] == '1':
                        ja += (1 - 2 * int(agent_decisions_decoded[asset_idx])) # multiply by sum(Uj) - Lj
                 utilidades_obs.append(utilidad_agente_1)

                

    # cos(Z) − exp(−Z/20) + epsilon_Y
    def Y(input_tensor, noise_mean=0, noise_stdev=0):
        input_tensor = input_tensor[..., :1]
        noise = torch.normal(noise_mean, noise_stdev, input_tensor.shape)
        return ((torch.cos(input_tensor)) - (torch.exp(-input_tensor / 20))) + noise
    
    def __init__(self, num_observations = 1000, num_objective_points = None):
        # By default, use double the number of observations to train the true model.
        if num_objective_points == None:
            num_objective_points = 2 * num_observations

        # Interventional domain
        self.interventional_domain = {'X': [-5,5], 'Z': [-5,20]}

        # Graph structure
        self.graph = SCM([('X', 'Z'), ('Z', 'Y')])

        # Same structure, deep copy
        self.true_graph = SCM([('U1', 'JA'), ('JA', 'Y')])

        # Generate observational data

        action_space = 7

        input_tensor_U1 = [randrange(action_space + 1) for i in range(0, num_observations)]
        input_tensor_U2 = [randrange(action_space + 1) for i in range(0, num_observations)]
        input_tensor_U3 = [randrange(action_space + 1) for i in range(0, num_observations)]
        input_tensor_U4 = [randrange(action_space + 1) for i in range(0, num_observations)]
        input_tensor_U5 = [randrange(action_space + 1) for i in range(0, num_observations)]
        input_tensor_U6 = [randrange(action_space + 1) for i in range(0, num_observations)]
        input_tensor_U7 = [randrange(action_space + 1) for i in range(0, num_observations)]
        input_tensor_U8 = [randrange(action_space + 1) for i in range(0, num_observations)]
        input_tensor_U9 = [randrange(action_space + 1) for i in range(0, num_observations)]
        input_tensor_U10 = [randrange(action_space + 1) for i in range(0, num_observations)]

        input_tensor_ja = torch.cat([torch.IntTensor(input_tensor_ui).view(-1,1) for input_tensor_ui in [input_tensor_U1,input_tensor_U2,input_tensor_U3,input_tensor_U4,input_tensor_U5,input_tensor_U6,input_tensor_U7,input_tensor_U8,input_tensor_U9,input_tensor_U10]], dim=1)
        obs_data_u1 = MAMGGraph.U1(torch.IntTensor(input_tensor_U1).view(-1,1), noise_stdev=None)
        obs_data_u2 = MAMGGraph.U2(torch.IntTensor(input_tensor_U2).view(-1,1), noise_stdev=None)
        obs_data_u3 = MAMGGraph.U3(torch.IntTensor(input_tensor_U3).view(-1,1), noise_stdev=None)
        obs_data_u4 = MAMGGraph.U4(torch.IntTensor(input_tensor_U4).view(-1,1), noise_stdev=None)
        obs_data_u5 = MAMGGraph.U5(torch.IntTensor(input_tensor_U5).view(-1,1), noise_stdev=None)
        obs_data_u6 = MAMGGraph.U6(torch.IntTensor(input_tensor_U6).view(-1,1), noise_stdev=None)
        obs_data_u7 = MAMGGraph.U7(torch.IntTensor(input_tensor_U7).view(-1,1), noise_stdev=None)
        obs_data_u8 = MAMGGraph.U8(torch.IntTensor(input_tensor_U8).view(-1,1), noise_stdev=None)
        obs_data_u9 = MAMGGraph.U9(torch.IntTensor(input_tensor_U9).view(-1,1), noise_stdev=None)
        obs_data_u10 = MAMGGraph.U10(torch.IntTensor(input_tensor_U10).view(-1,1), noise_stdev=None)
        obs_data_z = MAMGGraph.Z(obs_data_x, noise_stdev=1)
        obs_data_y = MAMGGraph.Y(obs_data_z, noise_stdev=1)

        # Add to dataframe
        self.observational_samples = DataFrame()
        self.observational_samples['X'] = torch.flatten(obs_data_x).tolist()
        self.observational_samples['Z'] = torch.flatten(obs_data_z).tolist()
        self.observational_samples['Y'] = torch.flatten(obs_data_y).tolist()
        # Shuffle dataframe into random order
        self.observational_samples.sample(frac=1)
        # Fit graph to observational data.
        self.graph.fit(self.observational_samples)

        # Generate objective data
        obs_data_x = ToyGraph.X(torch.linspace(-5, 5, num_objective_points).view(-1,1))
        obs_data_z = ToyGraph.Z(obs_data_x)
        obs_data_y = ToyGraph.Y(obs_data_z)

        # Add to dataframe
        self.objective_samples = DataFrame()
        self.objective_samples['X'] = torch.flatten(obs_data_x).tolist()
        self.objective_samples['Z'] = torch.flatten(obs_data_z).tolist()
        self.objective_samples['Y'] = torch.flatten(obs_data_y).tolist()

        # Fit graph to objective data.
        self.true_graph.fit(self.objective_samples)        

    # Wrapper for networkx draw()
    def draw(self):
        self.graph.draw()

    

        

    

