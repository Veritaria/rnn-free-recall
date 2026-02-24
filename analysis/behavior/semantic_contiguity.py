import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

from utils import savefig


class SemanticContiguity:
    def __init__(self) -> None:
        self.results = None
        self.results_all_time = None

    def fit(self, actions, similarity_matrix, action_value_max=64, bin_num=10):
        context_num = actions.shape[0]
        memory_num = actions.shape[1]
        
        similarities_for_nearby_recall = []
        for i in range(context_num):
            for t in range(memory_num-1):
                if actions[i][t] == action_value_max or actions[i][t+1] == action_value_max or actions[i][t] == actions[i][t+1]:
                    continue
                similarity = similarity_matrix[actions[i][t], actions[i][t+1]]
                similarities_for_nearby_recall.append(similarity)
        similarity_min = np.min(similarities_for_nearby_recall)
        similarity_max = np.max(similarities_for_nearby_recall)
        self.similarity_bins = np.linspace(similarity_min, similarity_max, bin_num+1)
        print(len(self.similarity_bins))
        self.results, edges = np.histogram(similarities_for_nearby_recall, bins=self.similarity_bins)
        print(self.results)

        result_baseline = []
        for k in range(5):
            action_permuted = actions.copy()
            for i in range(context_num):
                action_permuted[i] = np.random.permutation(action_permuted[i])
            similarities_for_nearby_recall = []
            for i in range(context_num):
                for t in range(memory_num-1):
                    if action_permuted[i][t] == action_value_max or action_permuted[i][t+1] == action_value_max or action_permuted[i][t] == action_permuted[i][t+1]:
                        continue
                    similarity = similarity_matrix[action_permuted[i][t], action_permuted[i][t+1]]
                    similarities_for_nearby_recall.append(similarity)
            baseline_histogram, _ = np.histogram(similarities_for_nearby_recall, bins=self.similarity_bins, density=False)
            result_baseline.append(baseline_histogram)
        self.results_baseline = np.mean(result_baseline, axis=0)
        print(self.results_baseline)
        self.results_baseline_to_divide = self.results_baseline.copy()
        self.results_baseline_to_divide[self.results_baseline_to_divide == 0] = 1
        self.results_normalized = self.results / self.results_baseline_to_divide
        self.results_normalized = self.results_normalized / np.sum(self.results_normalized)
        print(self.results_normalized)

        return self.results, self.results_baseline

    def visualize(self, save_path, save_name="all_time", use_normalized=False, title="", format="png"):
        if use_normalized:
            data = self.results_normalized
        else:
            data = self.results

        data = np.array(data, dtype=float)
        data[data == 0] = np.nan

        plt.figure(figsize=(4, 3.3), dpi=180)
        plt.scatter(self.similarity_bins[:-1], data, c='b', zorder=2)
        plt.plot(self.similarity_bins[:-1], data, c='k', zorder=1)
        plt.xlabel("semantic similarity")
        plt.ylabel("conditional\nrecall probability")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        savefig(save_path, save_name, format=format)

    def get_results(self):
        return self.results
  
