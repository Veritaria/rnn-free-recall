import numpy as np
from scipy.stats import rankdata


class TemporalFactor:
    def __init__(self) -> None:
        self.results = None

    def fit(self, memory_contexts, actions):
        self.context_num, self.memory_num = memory_contexts.shape
        self.results = np.zeros(memory_contexts.shape)
        for i in range(self.context_num):
            recalled = np.zeros(self.memory_num, dtype=bool)
            for t in range(len(actions[i]) - 1):
                # print("t:", t)
                position1 = np.where(memory_contexts[i] == actions[i][t])
                position2 = np.where(memory_contexts[i] == actions[i][t+1])
                # get a transition i->j that i and j are not wrong, not recalled and not the same
                if position1[0].shape[0] != 0 and position2[0].shape[0] != 0 and \
                    not recalled[position1[0][0]] and not recalled[position2[0][0]]\
                        and not position1[0][0] == position2[0][0]:
                    recalled[position1[0][0]] = True
                    # possible next memory to recall (delete all recalled ones)
                    possible_transitions = ~np.isin(memory_contexts[i], actions[i][:t+1])
                    # find the actual transition in the possible transitions
                    actual_transition = memory_contexts[i][possible_transitions] == actions[i][t+1]
                    # compute the distance of each possible transition to the current recalled item
                    possible_temporal_transitions = np.arange(self.memory_num)[possible_transitions] - position1[0][0]
                    if len(possible_temporal_transitions) > 1 and np.any(actual_transition):
                        # compute the rank number of all the possible transitions (the farer the transition is, the lower the rank is)
                        ranks = rankdata(-np.abs(possible_temporal_transitions))
                        # compute the rank of the actual transition / the largest rank
                        value = (ranks[actual_transition] - 1) / (len(ranks) - 1)
                        self.results[i][t+1] = value[0]
        return self.results


class TemporalFactorV2:
    def __init__(self) -> None:
        self.results = None

    def fit(self, memory_contexts, actions):
        self.context_num, self.memory_num = memory_contexts.shape
        self.results = np.zeros(memory_contexts.shape)
        self.results_mask = np.zeros(memory_contexts.shape)
        for i in range(self.context_num):
            recalled = np.zeros(self.memory_num, dtype=bool)
            for t in range(len(actions[i]) - 1):
                # print("t:", t)
                position1 = np.where(memory_contexts[i] == actions[i][t])
                position2 = np.where(memory_contexts[i] == actions[i][t+1])
                # get a transition i->j that i and j are not wrong, not recalled and not the same
                if position1[0].shape[0] != 0 and position2[0].shape[0] != 0 and \
                    not recalled[position1[0][0]] and not recalled[position2[0][0]]\
                        and not position1[0][0] == position2[0][0]:
                    # self.results_mask[i][t+1] = 1
                    recalled[position1[0][0]] = True
                    # possible next memory to recall (delete all recalled ones)
                    possible_transitions = ~np.isin(memory_contexts[i], actions[i][:t+1])
                    # find the actual transition in the possible transitions
                    actual_transition = memory_contexts[i][possible_transitions] == actions[i][t+1]
                    # compute the distance of each possible transition to the current recalled item
                    possible_temporal_transitions = np.arange(self.memory_num)[possible_transitions] - position1[0][0]
                    if len(possible_temporal_transitions) > 1 and np.any(actual_transition):
                        # compute the rank number of all the possible transitions (the farer the transition is, the lower the rank is)
                        ranks = rankdata(-np.abs(possible_temporal_transitions))
                        # compute the rank of the actual transition / the largest rank
                        value = (ranks[actual_transition] - 1) / (len(ranks) - 1)
                        self.results[i][t+1] = value[0]
                        self.results_mask[i][t+1] = 1
                    # elif len(possible_temporal_transitions) == 1 and np.any(actual_transition):
                    #     self.results[i][t+1] = 1
        # return self.results[:, 1:]
        return self.results[self.results_mask == 1]
    

if __name__ == "__main__":
    memory_contexts = np.array([[33,118,192,275,261,232,231,160,72,170,174,223,249,208,3]])
    actions = np.array([[118,160,3,249,231,28,170,135,208,284,0,0,0,0,0]])
    temporal_factor = TemporalFactor()
    results = temporal_factor.fit(memory_contexts, actions)
    print("results:", results)

