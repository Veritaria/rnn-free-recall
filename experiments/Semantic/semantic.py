
import os
import numpy as np
import sklearn.metrics.pairwise as skp
import matplotlib.pyplot as plt

from analysis.behavior import SemanticContiguity
from utils import savefig



def run(data_all, model_all, env, paths, exp_name, **kwargs):
    plt.rcParams['font.size'] = 18

    env = env[0]

    for run_name, data in data_all.items():
        run_name_without_num = run_name.split("-")[0]
        # fig_path = paths["fig"]/run_name
        run_num = run_name.split("-")[-1]
        fig_path = paths["fig"]/run_name_without_num/run_num
        fig_path.mkdir(parents=True, exist_ok=True)
        print()
        print(run_name)

        data = data[0]

        memory_num = env.unwrapped.sequence_len

        model = model_all[run_name]
        if hasattr(model, "step_for_each_timestep"):
            step_for_each_timestep = model.step_for_each_timestep
            timestep_each_phase = step_for_each_timestep * memory_num
        else:
            step_for_each_timestep = 1
            timestep_each_phase = memory_num
        # timestep_each_phase = env.memory_num

        # get recorded data and outputs of the model
        readouts = data['readouts']
        actions = data['actions']
        rewards = data['rewards']

        all_context_num = len(actions)
        context_num = min(all_context_num, 20)

        # convert data to numpy array
        memory_contexts = []
        for i in range(all_context_num):
            memory_contexts.append(data['trial_data'][i]["memory_sequence_int"])
        memory_contexts = np.array(memory_contexts)     # ground truth of memory for each trial
        # memory_contexts = memory_contexts.reshape(-1, memory_contexts.shape[-1])    # reshape to (trials, sequence_len)
        actions = np.array(actions).squeeze()                 # (trials, timesteps per trial)
        rewards = np.array(rewards)
        rewards = rewards.squeeze()
        rewards = rewards.reshape(-1, rewards.shape[-1])        # (trials, timesteps per trial)

        print(memory_contexts.shape, actions.shape, rewards.shape)



        """ semantic contiguity """
        memory_contexts = []
        for i in range(all_context_num):
            memory_contexts.append(data['trial_data'][i]["memory_sequence_int"])
        memory_contexts = np.array(memory_contexts)     # ground truth of memory for each trial
        memory_contexts = memory_contexts.reshape(-1, memory_contexts.shape[-1])    # reshape to (trials, sequence_len)
        print("memory_contexts shape: ", memory_contexts.shape)

        memory_contexts_features = []
        for i in range(all_context_num):
            memory_contexts_features.append(data['trial_data'][i]["memory_sequence"])
        memory_contexts_features = np.array(memory_contexts_features)
        print("memory_contexts_features shape: ", memory_contexts_features.shape)
        # Create permuted version of memory contexts by independently shuffling each trial's sequence
        memory_contexts_features_permuted = memory_contexts_features.copy()
        # for i in range(memory_contexts_features.shape[0]):
        #     perm = np.random.permutation(memory_contexts_features.shape[1])
        #     memory_contexts_features_permuted[i] = memory_contexts_features[i][perm]

        print(actions[:, :, :env.unwrapped.num_features].shape, memory_contexts_features_permuted.shape)
        print(actions[0, timestep_each_phase:, :env.unwrapped.num_features])
        print(memory_contexts_features_permuted[0])

        semantic_contiguity = SemanticContiguity()
        results = semantic_contiguity.fit(actions[:, timestep_each_phase:, :env.unwrapped.num_features], env.unwrapped.feature_dim)
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity_normalized", use_normalized=True, title="semantic contiguity", format="png")
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity", use_normalized=False, title="semantic contiguity", format="png")

        results_gt = semantic_contiguity.fit(memory_contexts_features_permuted, env.unwrapped.feature_dim)
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity_gt_normalized", use_normalized=True, title="semantic contiguity", format="png")
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity_gt", use_normalized=False, title="semantic contiguity", format="png")

        print(results, results_gt)
        # results_gt[results_gt == 0] = 1
        semantic_contiguity.results = results / results_gt
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity_norm_ratio", use_normalized=False, title="semantic contiguity", format="png")

        os.makedirs(fig_path/"data", exist_ok=True)
        np.save(fig_path/"data"/"semantic_contiguity_results.npy", results)
        np.save(fig_path/"data"/"semantic_contiguity_results_gt.npy", results_gt)
        np.save(fig_path/"data"/"semantic_contiguity_norm_ratio.npy", semantic_contiguity.results)


