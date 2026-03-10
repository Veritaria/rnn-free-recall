import os
import csv
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_mutual_info_score

from models.base_module import analyze
from train.criterions.rl import pick_action
from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import PCSelectivity, ItemIdentityDecoder, ItemIndexDecoder, Regressor, Classifier, MultiRegressor, CrossClassifier
from analysis.behavior import RecallProbability, RecallProbabilityInTime, TemporalFactor



def generate_data(dict_size, num_trials, num_items, timestep_each_phase):
    data = []
    rand_items = np.random.choice(dict_size, num_items, replace=False)
    for _ in range(num_trials):
        memory_sequence_index = np.random.choice(rand_items, timestep_each_phase, replace=False)
        data.append(memory_sequence_index)
    return data



def record_trials(model, env, data, timestep_each_phase):
    readouts = []
    trial_data = []
    actions = []
    for memory_sequence_index in data:
        obs_, info = env.reset(memory_sequence_index=memory_sequence_index)
        obs = torch.Tensor(obs_).reshape(1, -1)
        done = False
        model.reset_memory()
        state = model.init_state(1)

        actions_trial = []
        
        with analyze(model):
            memory_num = 0
            while not done:
                if info["phase"] == "encoding":
                    model.set_encoding(True)
                    model.set_retrieval(False)
                    memory_num += 1
                elif info["phase"] == "recall":
                    model.set_encoding(False)
                    model.set_retrieval(True)
                if "reset_state" in info and info["reset_state"]:
                    state = model.init_state(1, recall=True, prev_state=state)
                
                output, value, state, _ = model(obs, state)
                action_distribution = output
                action, log_prob_action, action_max = pick_action(action_distribution)
                obs_, reward, _, _, info = env.step(action_max.cpu().detach().numpy().squeeze(axis=1))
                done = info["done"]
                obs = torch.Tensor(obs_).reshape(1, -1)

                actions_trial.append(action_max.detach().cpu())

            readouts.append(model.readout())
            trial_data.append(env.unwrapped.get_trial_data())

            actions.append(np.stack(actions_trial))

    encoding_states = np.stack([readouts[i]['state'][:timestep_each_phase] for i in range(len(readouts))])
    recalling_states = np.stack([readouts[i]['state'][-timestep_each_phase:] for i in range(len(readouts))])
    memory_sequence = np.stack([trial_data[i]['memory_sequence_int'] for i in range(len(trial_data))])
    actions = np.stack(actions)

    encoding_states = encoding_states.squeeze()
    recalling_states = recalling_states.squeeze()
    actions = actions.squeeze()

    return encoding_states, recalling_states, memory_sequence, actions




def run(data_all, model_all, env, paths, exp_name, checkpoints=None, criterion=None):
    plt.rcParams['font.size'] = 16

    env = env[0]

    layer_names = ["encoder", "hidden", "decoder"]

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

        dict_size = env.unwrapped.vocabulary_size
        
        readouts = data['readouts']
        actions = data['actions']

        all_context_num = len(actions)


        test_memory_sequence = []
        for i in range(all_context_num):
            test_memory_sequence.append(data['trial_data'][i]["memory_sequence_int"])
        test_memory_sequence = np.array(test_memory_sequence)
        test_actions = np.array(actions).squeeze()
        # print(test_memory_sequence.shape, test_actions.shape)

        test_encoding_states = np.stack([readouts[i]['state'][1:timestep_each_phase] for i in range(all_context_num)]).squeeze()
        test_recalling_states = np.stack([readouts[i]['state'][-timestep_each_phase:-1] for i in range(all_context_num)]).squeeze()
        test_encoding_indexes = np.repeat(np.arange(sequence_len-1).reshape(1, -1), all_context_num, axis=0)
        test_recall_indexes = np.zeros_like(test_actions[:, -timestep_each_phase:])
        recall_index_mask = np.zeros_like(test_actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(sequence_len):
                if test_actions[i][-timestep_each_phase+t] in test_memory_sequence[i]:
                    recall_index_mask[i][t] = 1
                    test_recall_indexes[i][t] = np.where(test_memory_sequence[i] == test_actions[i][-timestep_each_phase+t])[0][0]
        recall_index_mask = recall_index_mask[:, :-1]
        test_recall_indexes = test_recall_indexes[:, :-1]
        print(test_encoding_states.shape, test_recalling_states.shape, test_encoding_indexes.shape, test_recall_indexes.shape, recall_index_mask.shape)
        data_encoding_test = test_encoding_states.reshape(-1, test_encoding_states.shape[-1])
        data_recall_test = test_recalling_states.reshape(-1, test_recalling_states.shape[-1])
        gt_encoding_test = test_encoding_indexes.reshape(-1)
        gt_recall_test = test_recall_indexes.reshape(-1)
        mask_test = recall_index_mask.reshape(-1)

        recording_trials = 1000

        decoding_accuracy_all = []
        cross_phase_decoding_accuracy_all = []
        for num_item in [8, 16, 32, 64]:
            decoding_accuracy = []
            cross_phase_decoding_accuracy = []
            for i in range(1):
                train_data = generate_data(dict_size, recording_trials, num_item, timestep_each_phase)
                train_encoding_states, train_recalling_states, train_memory_sequence, train_actions = record_trials(model, env, train_data, timestep_each_phase)
                # print(train_encoding_states.shape, train_recalling_states.shape, train_memory_sequence.shape, train_actions.shape)

                train_encoding_indexes = np.repeat(np.arange(sequence_len-1).reshape(1, -1), recording_trials, axis=0)
                train_recall_indexes = np.zeros_like(train_actions[:, -timestep_each_phase:])
                recall_index_mask = np.zeros_like(train_actions[:, -timestep_each_phase:], dtype=bool)
                for i in range(recording_trials):
                    for t in range(sequence_len):
                        if train_actions[i][-timestep_each_phase+t] in train_memory_sequence[i]:
                            recall_index_mask[i][t] = 1
                            train_recall_indexes[i][t] = np.where(train_memory_sequence[i] == train_actions[i][-timestep_each_phase+t])[0][0]
                recall_index_mask = recall_index_mask[:, :-1]
                train_recall_indexes = train_recall_indexes[:, :-1]

                train_encoding_states = train_encoding_states[:, 1:]
                train_recalling_states = train_recalling_states[:, :-1]

                # print(train_encoding_indexes[0], train_recall_indexes[0], recall_index_mask[0])
                # print(test_encoding_indexes[0], test_recall_indexes[0], recall_index_mask[0])

                # print(train_encoding_states.shape, train_recalling_states.shape, train_memory_sequence.shape, train_actions.shape)
            
                ridge_decoder = RidgeClassifier()
                data_encoding_train = train_encoding_states.reshape(-1, train_encoding_states.shape[-1])
                data_recall_train = train_recalling_states.reshape(-1, train_recalling_states.shape[-1])
                gt_encoding_train = train_encoding_indexes.reshape(-1)
                gt_recall_train = train_recall_indexes.reshape(-1)
                mask_train = recall_index_mask.reshape(-1)
                
                ridge_decoder.fit(data_encoding_train, gt_encoding_train)
                pred_encoding = ridge_decoder.predict(data_encoding_test)
                pred_cross_recall = ridge_decoder.predict(data_recall_test)
                encoding_accuracy = np.sum(pred_encoding == gt_encoding_test) / len(gt_encoding_test)
                cross_recall_accuracy = np.sum(pred_cross_recall == gt_recall_test) / len(gt_recall_test)
                # print(pred_encoding[0:10], gt_encoding_test[0:10], pred_cross_recall[0:10], gt_recall_test[0:10])
                
                ridge_decoder.fit(data_recall_train, gt_recall_train)
                pred_recall = ridge_decoder.predict(data_recall_test)
                pred_cross_encoding = ridge_decoder.predict(data_encoding_test)
                recall_accuracy = np.sum(pred_recall == gt_recall_test) / len(gt_recall_test)
                cross_encoding_accuracy = np.sum(pred_cross_encoding == gt_encoding_test) / len(gt_encoding_test)
                # print(pred_recall[0:10], gt_recall_test[0:10], pred_cross_encoding[0:10], gt_encoding_test[0:10])
                
                decoding_accuracy.append(np.mean([encoding_accuracy, recall_accuracy]))
                cross_phase_decoding_accuracy.append(np.mean([cross_recall_accuracy, cross_encoding_accuracy]))

            decoding_accuracy = np.array(decoding_accuracy)
            cross_phase_decoding_accuracy = np.array(cross_phase_decoding_accuracy)
            print(num_item, decoding_accuracy, cross_phase_decoding_accuracy)
            decoding_accuracy_all.append(decoding_accuracy)
            cross_phase_decoding_accuracy_all.append(cross_phase_decoding_accuracy)
            
        decoding_accuracy_all = np.array(decoding_accuracy_all)
        decoding_accuracy_all = np.mean(decoding_accuracy_all, axis=1)
        cross_phase_decoding_accuracy_all = np.array(cross_phase_decoding_accuracy_all)
        cross_phase_decoding_accuracy_all = np.mean(cross_phase_decoding_accuracy_all, axis=1)
        print(decoding_accuracy_all, cross_phase_decoding_accuracy_all)
        np.save(fig_path/"decoding_accuracy.npy", decoding_accuracy_all)
        np.save(fig_path/"cross_phase_decoding_accuracy.npy", cross_phase_decoding_accuracy_all)

        plt.figure(figsize=(4.3, 3.3), dpi=180)
        plt.plot(decoding_accuracy_all)
        plt.xlabel("number of items for training")
        plt.ylabel("decoding accuracy")
        plt.ylim(0, 1)
        ax = plt.gca()
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["8", "16", "32", "64"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        savefig(fig_path/"decoding_accuracy", "decoding_accuracy.png")


        plt.figure(figsize=(4.3, 3.3), dpi=180)
        plt.plot(cross_phase_decoding_accuracy_all)
        plt.xlabel("number of items for training")
        plt.ylabel("cross-phase decoding accuracy")
        plt.ylim(0, 1)
        ax = plt.gca()
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["8", "16", "32", "64"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        savefig(fig_path/"cross_phase_decoding_accuracy", "cross_phase_decoding_accuracy.png")



