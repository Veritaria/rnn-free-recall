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


        memory_sequence = []
        for i in range(all_context_num):
            memory_sequence.append(data['trial_data'][i]["memory_sequence_int"])
        memory_sequence = np.array(memory_sequence)
        actions = np.array(actions).squeeze()
        # print(memory_sequence.shape, actions.shape)

        encoding_states = np.stack([readouts[i]['state'][1:timestep_each_phase] for i in range(all_context_num)]).squeeze()
        recalling_states = np.stack([readouts[i]['state'][-timestep_each_phase:-1] for i in range(all_context_num)]).squeeze()
        recall_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(sequence_len):
                if actions[i][-timestep_each_phase+t] in memory_sequence[i]:
                    recall_mask[i][t] = 1
        recall_mask = recall_mask[:, :-1]
        memory_sequence = memory_sequence[:, 1:]
        actions = actions[:, -timestep_each_phase:-1]
        print(encoding_states.shape, recalling_states.shape, recall_mask.shape)


        test_encoding_states = encoding_states.reshape(-1, encoding_states.shape[-1])
        test_recall_states = recalling_states.reshape(-1, recalling_states.shape[-1])
        test_recall_mask = recall_mask.reshape(-1)
        gt_test_encoding_identity = memory_sequence.reshape(-1)
        gt_test_recall_identity = actions.reshape(-1)


        decoding_accuracy_all = []
        cross_phase_decoding_accuracy_all = []
        for num_timestep in [1, 2, 4, 7]:
            decoding_accuracy = []
            cross_phase_decoding_accuracy = []
            for i in range(5):
                rand_timestep = np.random.choice(np.arange(timestep_each_phase-1), num_timestep, replace=False)
                train_encoding_states = encoding_states[:, rand_timestep].reshape(-1, encoding_states.shape[-1])
                train_recalling_states = recalling_states[:, rand_timestep].reshape(-1, recalling_states.shape[-1])
                gt_train_encoding_identity = memory_sequence[:, rand_timestep].reshape(-1)
                gt_train_recall_identity = actions[:, rand_timestep].reshape(-1)
                train_recall_mask = recall_mask[:, rand_timestep].reshape(-1)
            
                ridge_decoder = RidgeClassifier()
                
                ridge_decoder.fit(train_encoding_states, gt_train_encoding_identity)
                pred_encoding = ridge_decoder.predict(test_encoding_states)
                pred_cross_recall = ridge_decoder.predict(test_recall_states[test_recall_mask])
                encoding_accuracy = np.sum(pred_encoding == gt_test_encoding_identity) / len(gt_test_encoding_identity)
                cross_recall_accuracy = np.sum(pred_cross_recall == gt_test_recall_identity[test_recall_mask]) / len(gt_test_recall_identity[test_recall_mask])
                # print(pred_encoding[0:10], gt_encoding_test[0:10], pred_cross_recall[0:10], gt_recall_test[0:10])
                
                ridge_decoder.fit(train_recalling_states[train_recall_mask], gt_train_recall_identity[train_recall_mask])
                pred_recall = ridge_decoder.predict(test_recall_states[test_recall_mask])
                pred_cross_encoding = ridge_decoder.predict(test_encoding_states)
                recall_accuracy = np.sum(pred_recall == gt_test_recall_identity[test_recall_mask]) / len(gt_test_recall_identity[test_recall_mask])
                cross_encoding_accuracy = np.sum(pred_cross_encoding == gt_test_encoding_identity) / len(gt_test_encoding_identity)
                # print(pred_recall[0:10], gt_recall_test[0:10], pred_cross_encoding[0:10], gt_encoding_test[0:10])
                
                decoding_accuracy.append(np.mean([encoding_accuracy, recall_accuracy]))
                cross_phase_decoding_accuracy.append(np.mean([cross_recall_accuracy, cross_encoding_accuracy]))

            decoding_accuracy = np.array(decoding_accuracy)
            cross_phase_decoding_accuracy = np.array(cross_phase_decoding_accuracy)
            print(num_timestep, decoding_accuracy, cross_phase_decoding_accuracy)
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
        plt.xlabel("number of timesteps for training")
        plt.ylabel("decoding accuracy")
        plt.ylim(0, 1)
        ax = plt.gca()
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        savefig(fig_path/"decoding_accuracy", "decoding_accuracy.png")


        plt.figure(figsize=(4.3, 3.3), dpi=180)
        plt.plot(cross_phase_decoding_accuracy_all)
        plt.xlabel("number of timesteps for training")
        plt.ylabel("cross-phase decoding accuracy")
        plt.ylim(0, 1)
        ax = plt.gca()
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        savefig(fig_path/"cross_phase_decoding_accuracy", "cross_phase_decoding_accuracy.png")



