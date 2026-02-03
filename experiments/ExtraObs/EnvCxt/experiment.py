import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
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
                print("context {}, gt: {}, action: {}, retrieved memory: {}, rewards: {}".format(i, memory_contexts[i], actions[i][sequence_len:], 
                np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(), axis=1)+1, rewards[i][sequence_len:]))
            else:
                print("context {}, gt: {}, action: {}, rewards: {}".format(i, memory_contexts[i], actions[i][sequence_len:], 
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

        """ memory gate """
        # if "mem_gate_recall" in readouts[0]:
        #     plt.figure(figsize=(4, 3), dpi=180)
        #     for i in range(context_num):
        #         em_gates = readouts[i]['mem_gate_recall']
        #         plt.plot(np.mean(em_gates.squeeze(1), axis=-1)[:timestep_each_phase], label="context {}".format(i))
        #     ax = plt.gca()
        #     ax.spines['top'].set_visible(False)
        #     ax.spines['right'].set_visible(False)
        #     plt.xlabel("time of recall phase")
        #     plt.ylabel("memory gate")
        #     plt.tight_layout()
        #     savefig(fig_path, "em_gate_recall")

        """ recall probability (output) (CRP curve) """
        recall_probability = RecallProbability()
        recall_probability.fit(memory_contexts, actions[:, -timestep_each_phase:])
        # plot CRP curve
        recall_probability.visualize_all_time(fig_path/"recall_prob", format="svg")
        recall_probability.visualize(fig_path/"recall_prob", format="svg")
        results_all_time = recall_probability.get_results_all_time()
        # write to csv file
        with open(fig_path/"recall_probability.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(results_all_time)

        """ count temporal factor and forward asymmetry """
        recall_probability = RecallProbability()
        recall_probability.fit(memory_contexts, actions[:, -timestep_each_phase:])
        forward_asymmetry = recall_probability.forward_asymmetry
        temporal_factor = TemporalFactor()
        temp_fact = temporal_factor.fit(memory_contexts, actions[:, -timestep_each_phase:])
        temp_fact = np.mean(temp_fact)
        print("forward asymmetry:[{},{}]".format(data['accuracy'], forward_asymmetry))
        print("temporal factor:[{},{}]".format(data['accuracy'], temp_fact))
        # write to csv file
        with open(fig_path/"contiguity_effect.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow([data['accuracy'], forward_asymmetry, temp_fact])

        """ recall probability of first timestep (see primacy and recency) """
        recall_probability_in_time = RecallProbabilityInTime()
        recall_probability_in_time.fit(memory_contexts, actions[:, -timestep_each_phase:])
        recall_probability_in_time.visualize(fig_path)



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




        """ decode item identity """
        retrieved_memories = []
        for i in range(all_context_num):
            retrieved_memory = readouts[i]["ValueMemory"]["similarity"].squeeze()
            retrieved_memory = np.argmax(retrieved_memory, axis=-1)
            retrieved_memories.append(retrieved_memory)
        retrieved_memories = np.stack(retrieved_memories)

        c_memorizing = np.stack([readouts[i]['state'][:timestep_each_phase].squeeze() for i in range(all_context_num)])   # context_num * time * state_dim
        c_recalling = np.stack([readouts[i]['state'][-timestep_each_phase:].squeeze() for i in range(all_context_num)])
        memory_sequence = np.stack([memory_contexts[i] for i in range(all_context_num)]) - 1    # context_num * time

        # Ridge
        ridge_decoder = RidgeClassifier()
        ridge = ItemIdentityDecoder(decoder=ridge_decoder)
        ridge_encoding_res, ridge_encoding_stat_res = ridge.fit(c_memorizing.transpose(1, 0, 2), memory_sequence.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase")
        np.save(fig_path/"ridge_encoding.npy", ridge_encoding_res)
        # np.save(fig_path/"ridge_encoding_stat.npy", list(ridge_encoding_stat_res.values()))

        ridge_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(sequence_len):
                if actions[i][-timestep_each_phase+t] in memory_contexts[i]:
                    ridge_mask[i][t] = 1
        ridge_recall_res, ridge_recall_stat_res = ridge.fit(c_recalling.transpose(1, 0, 2), actions[:, -timestep_each_phase:].transpose(1, 0), ridge_mask.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec", colormap_label="item position\nin recall order",
                                xlabel="time in recall phase")
        np.save(fig_path/"ridge_recall.npy", ridge_recall_res)


        """ decode item index """
        encoding_index = np.repeat(np.arange(sequence_len).reshape(1, -1), all_context_num, axis=0)

        recall_index = np.zeros_like(actions[:, -timestep_each_phase:])
        index_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(sequence_len):
                if actions[i][-timestep_each_phase+t] in memory_contexts[i]:
                    index_mask[i][t] = 1
                    recall_index[i][t] = np.where(memory_contexts[i] == actions[i][-timestep_each_phase+t])[0][0]

        # Ridge
        ridge_decoder = RidgeClassifier()
        ridge = ItemIndexDecoder(decoder=ridge_decoder)
        ridge_encoding_res, index_encoding_acc, index_encoding_r2 = ridge.fit(c_memorizing, encoding_index)
        ridge.visualize(save_path=fig_path/"ridge_index", save_name="c_enc", xlabel="time in encoding phase")
        np.save(fig_path/"ridge_encoding_index.npy", ridge_encoding_res)

        ridge_recall_res, index_recall_acc, index_recall_r2 = ridge.fit(c_recalling, recall_index, index_mask)
        ridge.visualize(save_path=fig_path/"ridge_index", save_name="c_rec", xlabel="time in recall phase")
        np.save(fig_path/"ridge_recall_index.npy", ridge_recall_res)

        ridge_classifier_stat = {
            "item_enc_acc": ridge_encoding_stat_res["acc"],
            "item_enc_r2": ridge_encoding_stat_res["r2"],
            "item_enc_acc_last": ridge_encoding_stat_res["acc_last"],
            "item_enc_r2_last": ridge_encoding_stat_res["r2_last"],
            "item_rec_acc": ridge_recall_stat_res["acc"],
            "item_rec_r2": ridge_recall_stat_res["r2"],
            "item_rec_acc_last": ridge_recall_stat_res["acc_last"],
            "item_rec_r2_last": ridge_recall_stat_res["r2_last"],
            "index_enc_acc": index_encoding_acc,
            "index_enc_r2": index_encoding_r2,
            "index_rec_acc": index_recall_acc,
            "index_rec_r2": index_recall_r2
        }
        with open(fig_path/"ridge_classifier_stat.pkl", "wb") as f:
            pickle.dump(ridge_classifier_stat, f)



        """ explained variance of index and identity """
        multi_regressor = MultiRegressor()
        r2_index_encoding, r2_identity_encoding = multi_regressor.fit(c_memorizing, encoding_index, memory_sequence)
        print("r2_index_encoding: ", r2_index_encoding)
        print("r2_identity_encoding: ", r2_identity_encoding)
        r2_index_recall, r2_identity_recall = multi_regressor.fit(c_recalling, recall_index, actions[:, -timestep_each_phase:], ridge_mask)
        print("r2_index_recall: ", r2_index_recall)
        print("r2_identity_recall: ", r2_identity_recall)

        # plot the explained variance of index and identity
        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.bar(["index", "identity"], [r2_index_encoding, r2_identity_encoding])
        plt.xlabel("variable")
        plt.ylabel("explained variance")
        plt.tight_layout()
        savefig(fig_path/"multi_regression", "explained_variance_encoding")

        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.bar(["index", "identity"], [r2_index_recall, r2_identity_recall])
        plt.xlabel("variable")
        plt.ylabel("explained variance")
        plt.tight_layout()
        savefig(fig_path/"multi_regression", "explained_variance_recall")

        # plot encoding and recall phases together
        plt.figure(figsize=(4.5, 3.7), dpi=180)
        bar_width = 0.35
        index = np.arange(2)
        
        plt.bar(index, [r2_index_encoding, r2_index_recall], bar_width, label="index")
        plt.bar(index + bar_width, [r2_identity_encoding, r2_identity_recall], bar_width, label="identity")
        
        plt.xlabel("variable")
        plt.ylabel("explained variance")
        plt.xticks(index + bar_width / 2, ["encoding", "recall"])
        plt.legend()
        plt.tight_layout()
        savefig(fig_path/"multi_regression", "explained_variance_combined")


        # put encoding and recall data together
        # only consider time steps 2-8
        c_all = np.concatenate([c_memorizing[:, 1:8, :], c_recalling[:, :-1, :]], axis=0)
        index_all = np.concatenate([encoding_index[:, 1:8], recall_index[:, 1:8]], axis=0)
        memory_sequence_all = np.concatenate([memory_sequence[:, 1:8]+1, actions[:, -timestep_each_phase:-1]], axis=0)
        print(c_all.shape, index_all.shape, memory_sequence_all.shape)

        multi_regressor = MultiRegressor()
        r2_index_all, r2_identity_all = multi_regressor.fit(c_all, index_all, memory_sequence_all)
        print("r2_index_all: ", r2_index_all)
        print("r2_identity_all: ", r2_identity_all)

        plt.figure(figsize=(3, 3.7), dpi=180)
        plt.bar(["index", "identity"], [r2_index_all, r2_identity_all])
        plt.xlabel("variable")
        plt.ylabel("explained variance")
        plt.tight_layout()
        savefig(fig_path/"multi_regression", "explained_variance_combined_all")

        explained_variance = np.stack([r2_index_encoding, r2_index_recall, r2_identity_encoding, r2_identity_recall, r2_index_all, r2_identity_all])
        np.save(fig_path/"explained_variance.npy", explained_variance)



        """ cross-phase classification """
        # index
        c_recalling_for_index = np.stack([readouts[i]['state'][timestep_each_phase:timestep_each_phase*2].squeeze() for i in range(all_context_num)])
        cross_classifier = CrossClassifier()
        cross_classifier.fit(c_memorizing, encoding_index)
        r2_index_rec, acc_index_rec = cross_classifier.score(c_recalling_for_index, recall_index, index_mask)
        cross_classifier.fit(c_recalling_for_index, recall_index, index_mask)
        r2_index_enc, acc_index_enc = cross_classifier.score(c_memorizing, encoding_index)
        print("cross acc_index_enc, acc_index_rec: ", acc_index_enc, acc_index_rec)

        # identity
        cross_classifier = CrossClassifier()
        # print(memory_sequence[:5]+1, actions[:5, -timestep_each_phase:])
        cross_classifier.fit(c_memorizing, memory_sequence+1)
        r2_identity_rec, acc_identity_rec = cross_classifier.score(c_recalling, actions[:, -timestep_each_phase:], ridge_mask)
        cross_classifier.fit(c_recalling, actions[:, -timestep_each_phase:], ridge_mask)
        r2_identity_enc, acc_identity_enc = cross_classifier.score(c_memorizing, memory_sequence+1)
        print("cross acc_identity_enc, acc_identity_rec: ", acc_identity_enc, acc_identity_rec)


        plt.figure(figsize=(4.5, 3.7), dpi=180)
        bar_width = 0.35
        index = np.arange(2)
        
        plt.bar(index, [acc_index_enc, acc_identity_enc], bar_width, label="recall-encoding")
        plt.bar(index + bar_width, [acc_index_rec, acc_identity_rec], bar_width, label="encoding-recall")
        
        plt.xlabel("variable")
        plt.ylabel("decoding accuracy")
        plt.xticks(index + bar_width / 2, ["index", "identity"])
        plt.legend()
        plt.tight_layout()
        savefig(fig_path/"cross_classification", "cross_phase_accuracy")

        cross_acc = np.stack([acc_index_enc, acc_identity_enc, acc_index_rec, acc_identity_rec])
        np.save(fig_path/"cross_acc.npy", cross_acc)



        """ PC selectivity """
        # # convert actions and item index to one-hot
        # actions_one_hot = np.zeros((all_context_num, sequence_len, env.vocabulary_num))
        # for i in range(all_context_num):
        #     actions_one_hot[i] = np.eye(env.vocabulary_num)[actions[i, sequence_len:]-1]

        # # memory content
        # memories_one_hot = np.zeros((all_context_num, sequence_len, env.vocabulary_num))
        # for i in range(all_context_num):
        #     memories_one_hot[i] = np.eye(env.vocabulary_num)[memory_contexts[i]-1]

        # # memory index
        # # print(retrieved_memories.shape)
        # retrieved_memories_one_hot = np.zeros((all_context_num, sequence_len, sequence_len))
        # for i in range(all_context_num):
        #     retrieved_memories_one_hot[i] = np.eye(sequence_len)[retrieved_memories[i]]
        
        # # labels = {"actions": actions[:, sequence_len:], "memory index": retrieved_memories}

        # pc_selectivity = PCSelectivity(n_components=128, reg=RidgeClassifier())
        # # labels = {"memory content": memories_one_hot, "memory index": retrieved_memories_one_hot}
        # labels = {"item identity": memory_contexts-1, "item index": retrieved_memories}
        # selectivity, explained_var = pc_selectivity.fit(c_memorizing, labels)
        # pc_selectivity.visualize(save_path=fig_path/"pc_selectivity", file_name="encoding", format="svg")
        # np.savez(fig_path/"pc_selectivity_encoding.npz", selectivity=selectivity, explained_var=explained_var, labels=labels)

        # pc_selectivity = PCSelectivity(n_components=128, reg=RidgeClassifier())
        # # labels = {"recalled memory": actions_one_hot, "memory index": retrieved_memories_one_hot}
        # labels = {"item identity": actions[:, sequence_len:]-1, "item index": retrieved_memories}
        # selectivity, explained_var = pc_selectivity.fit(c_recalling, labels)
        # pc_selectivity.visualize(save_path=fig_path/"pc_selectivity", file_name="recalling", format="svg")
        # np.savez(fig_path/"pc_selectivity_recalling.npz", selectivity=selectivity, explained_var=explained_var, labels=labels)


        """ policy distribution over all memory items """
        # policy = []
        # for i in range(all_context_num):
        #     policy.append(readouts[i]['decision'])
        # policy = np.stack(policy).squeeze()
        # print(policy.shape, actions.shape)

        # policy_sorted = []
        # for i in range(all_context_num):
        #     policy_sorted.append(policy[i, -sequence_len:, actions[i, -sequence_len:]])
        # policy_sorted = np.stack(policy_sorted).squeeze()
        # print(policy_sorted.shape)

        # plt.imshow(policy_sorted[0], cmap="Blues")
        # plt.colorbar()
        # plt.title("policy distribution, one trial")
        # plt.xlabel("time step")
        # plt.ylabel("memory item")
        # plt.tight_layout()
        # savefig(fig_path/"policy", "one_trial.png")

        # plt.imshow(np.mean(policy_sorted, axis=0), cmap="Blues")
        # plt.colorbar()
        # plt.title("policy distribution, averaged")
        # plt.xlabel("time step")
        # plt.ylabel("memory item")
        # plt.tight_layout()
        # savefig(fig_path/"policy", "all_trial.png")



        """ do the hidden state get away from the just recalled item? """
        # get all the data needed
        rec_states = []
        retrieved_memories = []
        most_similar_memories = []
        for i in range(all_context_num):
            state = readouts[i]['state'].squeeze()
            rec_states.append(state[-timestep_each_phase:])
            retrieved_memories.append(readouts[i]['ValueMemory']['retrieved_memory'].squeeze()[-timestep_each_phase:])
            memory_similarity = readouts[i]['ValueMemory']['similarity']
            most_similar_index = np.argmax(memory_similarity, axis=-1).squeeze()[-timestep_each_phase:]
            most_similar_memories.append(state[most_similar_index])
        rec_states = np.stack(rec_states)
        retrieved_memories = np.stack(retrieved_memories)
        most_similar_memories = np.stack(most_similar_memories)
        print("rec_states shape: ", rec_states.shape)
        print("retrieved_memories shape: ", retrieved_memories.shape)

        # calculate the distance between the hidden state and the just recalled memory
        distances = np.zeros((timestep_each_phase, timestep_each_phase))
        for i in range(timestep_each_phase):
            for j in range(timestep_each_phase):
                dist = 0.0
                for k in range(all_context_num):
                    x = rec_states[k][i] / np.linalg.norm(rec_states[k][i])
                    y = retrieved_memories[k][j] / np.linalg.norm(retrieved_memories[k][j])
                    # print(x.shape, y.shape)
                    if np.sum(x * y) > 1:
                        print("strange cosine similarity: ", np.sum(x * y))
                    dist += np.sum(x * y)
                distances[i][j] = dist / all_context_num
        # distances = distances / np.sum(np.abs(distances), axis=-1, keepdims=True)
        
        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.imshow(distances, cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(label="cosine similarity")
        plt.xlabel("retrieved memory")
        plt.ylabel("time in recall phase")
        plt.tight_layout()
        savefig(fig_path/"distances", "state_retrieved_memory.png")


        # calculate the distance between the hidden state and the most similar memory
        distances = np.zeros((timestep_each_phase, timestep_each_phase))
        for i in range(timestep_each_phase):
            for j in range(timestep_each_phase):
                dist = 0.0
                for k in range(all_context_num):
                    x = rec_states[k][i] / np.linalg.norm(rec_states[k][i])
                    y = most_similar_memories[k][j] / np.linalg.norm(most_similar_memories[k][j])
                    if np.sum(x * y) > 1:
                        print("strange cosine similarity: ", np.sum(x * y))
                    dist += np.sum(x * y)
                distances[i][j] = dist / all_context_num
        # distances = distances / np.sum(np.abs(distances), axis=-1, keepdims=True)
        
        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.imshow(distances, cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(label="cosine similarity")
        plt.xlabel("most similar memory")
        plt.ylabel("time in recall phase")
        plt.tight_layout()
        savefig(fig_path/"distances", "state_most_similar_memory.png")



        """ is there a bias to recall a particular item at a particular time step? """
        vocabulary_size = env.unwrapped.vocabulary_size
        recall_num_by_time = np.zeros((timestep_each_phase, vocabulary_size+1))
        rec_actions = actions[:, -timestep_each_phase:].astype(int)
        for i in range(context_num):
            for t in range(timestep_each_phase):
                recall_num_by_time[t][rec_actions[i][t]] += 1
        plt.figure(figsize=(0.25*vocabulary_size, 3.7), dpi=180)
        plt.imshow(recall_num_by_time, cmap="Blues")
        plt.colorbar(label="recall number")
        plt.xlabel("memory item")
        plt.ylabel("time in recall phase")
        plt.tight_layout()
        savefig(fig_path/"recall_num_by_time", "recall_num_by_time.png")



