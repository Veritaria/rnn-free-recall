
import os
import numpy as np
import sklearn.metrics.pairwise as skp
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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


        if env.unwrapped.extra_observation_type == "hierarchical_binary":
            semantic_stimuli = env.unwrapped.hierarchical_binary_data
        elif env.unwrapped.extra_observation_type == "gaussian_identity":
            semantic_stimuli = env.unwrapped.gaussian_vecs
        else:
            raise ValueError(f"Invalid extra observation type for this analysis: {env.unwrapped.extra_observation_type}")

        similarity_matrix = skp.cosine_similarity(semantic_stimuli)

        plt.figure(figsize=(4.5, 3.7), dpi=180)
        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1) 

        plt.imshow(similarity_matrix, cmap="RdYlBu_r", norm=norm)
        plt.colorbar(label="cosine similarity")
        plt.xlabel("semantic stimulus")
        plt.ylabel("semantic stimulus")
        plt.tight_layout()
        savefig(fig_path, "semantic_stimuli_similarity")


        semantic_contiguity = SemanticContiguity()
        results, results_baseline = semantic_contiguity.fit(actions[:, timestep_each_phase:], similarity_matrix)
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity_normalized", use_normalized=True, )
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity", use_normalized=False, )


        os.makedirs(fig_path/"data", exist_ok=True)
        np.save(fig_path/"data"/"semantic_contiguity_results.npy", results)
        np.save(fig_path/"data"/"semantic_contiguity_baseline.npy", results_baseline)


