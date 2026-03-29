import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from train.criterions.rl import pick_action
from models.utils import entropy
from utils import savefig


def generate_data(env, num_trials):
    """
    randomly generate the memory sequence index for a bunch of trials
    """
    data = []
    for _ in range(num_trials):
        env.reset()
        data.append(env.unwrapped.memory_sequence_index)
    return data


def record_performance(model, env, criterion, data): 
    # Compute for each data point
    correct_actions, wrong_actions, not_know_actions = 0, 0, 0
    for sequence in data:
        # Reset environment and model state
        obs_, info =env.reset(memory_sequence_index=sequence)
        obs = torch.Tensor(obs_).reshape(1, -1)
        done = False
        model.reset_memory()
        state = model.init_state(1)

        memory_num = 0
        # actions = []
        while not done:
            # set up the phase of the model 
            if info["phase"] == "encoding":
                model.set_encoding(True)
                model.set_retrieval(False)
                memory_num += 1
            elif info["phase"] == "recall":
                model.set_encoding(False)
                model.set_retrieval(True)
            # reset state between phases
            if "reset_state" in info and info["reset_state"]:
                state = model.init_state(1, recall=True, prev_state=state)
            
            output, value, state, _ = model(obs, state)
            action_distribution = output
            action, log_prob_action, action_max = pick_action(action_distribution)
            obs_, reward, _, _, info = env.step(action_max.cpu().detach().numpy().squeeze(axis=1))
            obs = torch.Tensor(obs_).reshape(1, -1)

            # actions.append(action_max)
            correct_actions += np.sum(info["correct"])
            wrong_actions += np.sum(info["wrong"])
            not_know_actions += np.sum(info["not_know"])

            done = info["done"]

        #     print(info)

        # print(env.unwrapped.memory_sequence_index, actions[memory_num:])        
        # print(correct_actions, wrong_actions, not_know_actions)

    accuracy = correct_actions / (correct_actions + wrong_actions + not_know_actions)
    
    return accuracy


def analyze_noise_perturbation(model, env, criterion, noise_levels, save_path):
    """Analyze whether model is in rich or lazy regime by tracking NTK over training"""

    accuracies = []
    
    # Create save directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    data = generate_data(env, 100)
    
    # Track changes during training
    for i, noise_level in enumerate(noise_levels):
        # x_batch, _ = next(iter(dataloader))
        model.wm_noise_prop = noise_level
        
        accuracy = record_performance(model, env, criterion, data)
        accuracies.append(accuracy)

        print("noise level", noise_level, "accuracy", accuracy)

    accuracies = np.array(accuracies)

    # Save numerical results
    np.save(save_path / 'perturbation_accuracies.npy', accuracies)

    return accuracies


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

        model = model_all[run_name]
        

        noise_levels = np.arange(0, 1.01, 0.05)
        accuracies = analyze_noise_perturbation(model, env, criterion, noise_levels, fig_path)

        plt.figure(figsize=(5, 4), dpi=180)
        plt.plot(noise_levels, accuracies)
        plt.xticks(rotation=45)
        plt.xlabel('Noise Level')
        plt.ylabel('Accuracy')
        ax = plt.gca()
        ax.set_ylim(0, 1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        savefig(fig_path / "perturbation", "perturbation_accuracies")
        plt.close()
