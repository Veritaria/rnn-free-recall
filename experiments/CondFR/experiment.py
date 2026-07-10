import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_mutual_info_score

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import PCSelectivity, ItemIdentityDecoder, ItemIndexDecoder, Regressor, Classifier, MultiRegressor, CrossClassifier
from analysis.behavior import RecallProbability, RecallProbabilityInTime, TemporalFactor



def run(data_all, model_all, env, paths, exp_name, checkpoints=None, **kwargs):
    plt.rcParams['font.size'] = 16

    mpl.rcParams['svg.fonttype'] = 'none'

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
        model = model_all[run_name]

        sequence_len = env.unwrapped.sequence_len
        if hasattr(model, "step_for_each_timestep"):
            step_for_each_timestep = model.step_for_each_timestep
            timestep_each_phase = step_for_each_timestep * sequence_len
        else:
            step_for_each_timestep = 1
            timestep_each_phase = sequence_len

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

        if "ValueMemory" in readouts[0] and "similarity" in readouts[0]["ValueMemory"]:
            has_memory = True
        else:
            has_memory = False
        
        # print ground truths, actions and rewards for 5 trials
        print("accuracy: {}".format(data['accuracy']))
        for i in range(5):
            if has_memory:
                print("context {}, gt: {}, action: {}, retrieved memory: {}, rewards: {}".format(i, memory_contexts[i][data['trial_data'][i]["matched_item_indexes"]], actions[i][sequence_len:], 
                np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(), axis=1)+1, rewards[i][sequence_len:]))
            else:
                print("context {}, gt: {}, action: {}, rewards: {}".format(i, memory_contexts[i][data['trial_data'][i]["matched_item_indexes"]], actions[i][sequence_len:], 
                rewards[i][sequence_len:]))

        """ similarity of states """
        similarities = []
        for i in range(all_context_num):
            states = readouts[i]["state"].squeeze()
            similarity = skp.cosine_similarity(states, states)
            similarities.append(similarity)
        similarities = np.stack(similarities)
        similarity = np.mean(similarities, axis=0)

        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.imshow(similarity[timestep_each_phase:timestep_each_phase*2, :timestep_each_phase], cmap="Blues")
        plt.colorbar(label="cosine similarity\nbetween hidden states")
        plt.xlabel("time in encoding phase")
        plt.ylabel("time in recall phase")
        # set the color bar to be between 0 and 1
        plt.clim(0, 1)  # set color limits to [0, 1]
        # plt.title("encoding-recalling state similarity")
        plt.tight_layout()
        savefig(fig_path/"state_similarity", "encode_recall")

        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.imshow(similarity[:timestep_each_phase, :timestep_each_phase], cmap="Blues")
        plt.colorbar(label="cosine similarity\nbetween hidden states")
        plt.xlabel("time in encoding phase")
        plt.ylabel("time in encoding phase")
        plt.clim(0, 1)
        plt.tight_layout()
        savefig(fig_path/"state_similarity", "encode_encode")

        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.imshow(similarity[timestep_each_phase:timestep_each_phase*2, timestep_each_phase:timestep_each_phase*2], cmap="Blues")
        plt.colorbar(label="cosine similarity\nbetween hidden states")
        plt.xlabel("time in recall phase")
        plt.ylabel("time in recall phase")
        plt.clim(0, 1)
        plt.tight_layout()
        savefig(fig_path/"state_similarity", "recall_recall")


        """ recall probability (output) (CRP curve) """
        # delete all unmatched items in memory context and all zeros in actions
        memory_contexts_matched = np.zeros((all_context_num, sequence_len))
        actions_matched = np.zeros((all_context_num, timestep_each_phase))
        for i in range(all_context_num):
            action_list = actions[i, sequence_len:][actions[i, sequence_len:] != 0]
            actions_matched[i][:len(action_list)] = action_list
            memory_matched = data['trial_data'][i]["memory_sequence_int"][data['trial_data'][i]["matched_item_indexes"]]
            memory_contexts_matched[i][:len(memory_matched)] = memory_matched
        
        recall_probability = RecallProbability()
        recall_probability.fit(memory_contexts_matched, actions_matched)
        # plot CRP curve
        recall_probability.visualize_all_time(fig_path/"recall_prob", format="svg")
        recall_probability.visualize(fig_path/"recall_prob", format="svg")
        results_all_time = recall_probability.get_results_all_time()
        # write to csv file
        with open(fig_path/"recall_probability.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(results_all_time)


        retrieved_memories = []
        memory_gates = []
        for i in range(all_context_num):
            retrieved_memory = readouts[i]["ValueMemory"]["similarity"].squeeze()
            retrieved_memory = np.argmax(retrieved_memory, axis=-1)
            retrieved_memories.append(retrieved_memory)
            memory_gates.append(readouts[i]['mem_gate_recall'].squeeze())
        retrieved_memories = np.stack(retrieved_memories)
        memory_gates = np.stack(memory_gates)
        print("retrieved_memories shape: ", retrieved_memories.shape)
        print("memory_gates shape: ", memory_gates.shape)
        """ recall probability (retrieved memory) """
        recall_probability = RecallProbability()
        recall_probability.fit(np.repeat(np.arange(timestep_each_phase).reshape(1,-1), retrieved_memories.shape[0], axis=0), retrieved_memories)
        recall_probability.visualize(fig_path/"recall_prob_memory", format="png")
        recall_probability.visualize_all_time(fig_path/"recall_prob_memory", format="png")

        """ number of unique memories retrieved and probability of retrieving each memory """
        num_unique_memories = []
        num_retrieve_memory = np.zeros(timestep_each_phase)
        num_retrieve_memory_by_time = np.zeros(timestep_each_phase)
        for i in range(all_context_num):
            num_unique_memory = 0
            retrieved = np.zeros(timestep_each_phase, dtype=bool)
            for t in range(timestep_each_phase):
                if not retrieved[retrieved_memories[i][t]]:
                    retrieved[retrieved_memories[i][t]] = True
                    num_unique_memory += 1
                    num_retrieve_memory[retrieved_memories[i][t]] += 1
                    num_retrieve_memory_by_time[t] += memory_gates[i][t]
            num_unique_memories.append(num_unique_memory)
        num_unique_memories = np.array(num_unique_memories)
        prob_retrieve_memory = num_retrieve_memory / all_context_num
        prob_retrieve_memory_by_time = num_retrieve_memory_by_time / all_context_num

        # plot the distribution
        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.hist(num_unique_memories, bins=np.arange(0, timestep_each_phase+1, 1), edgecolor="black")
        plt.xlabel("number of unique\nmemories retrieved")
        plt.ylabel("proportion of trials")
        plt.tight_layout()
        savefig(fig_path/"num_retrieve_memory", "num_unique_memories")

        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.bar(np.arange(1, timestep_each_phase+1), prob_retrieve_memory)
        plt.xlabel("memory index")
        plt.ylabel("probability of\nretrieving memory")
        plt.tight_layout()
        savefig(fig_path/"num_retrieve_memory", "prob_retrieve_each_memory")

        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.bar(np.arange(1, timestep_each_phase+1), prob_retrieve_memory_by_time)
        plt.xlabel("time step in recall phase")
        plt.ylabel("probability of\nretrieving memory")
        plt.tight_layout()
        savefig(fig_path/"num_retrieve_memory", "prob_retrieve_each_timestep")




        states = []
        for i in range(all_context_num):
            states.append(readouts[i]['state'])
        states = np.stack(states).squeeze()
        print(states.shape)
        
        """ PCA """
        pca = PCA()
        pca.fit(states)
        pca.visualize_state_space(trial_num=20, save_path=fig_path/"pca_state_space", end_step=timestep_each_phase, colormap_label="time in\nencoding phase", 
                                file_name="encoding", format="svg")
        pca.visualize_state_space(trial_num=20, save_path=fig_path/"pca_state_space", start_step=timestep_each_phase, end_step=timestep_each_phase*2,
                                colormap_label="time in recall phase", file_name="recall", format="svg")

