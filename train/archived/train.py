import time
from datetime import timedelta
from collections import defaultdict
import numpy as np
import torch

from .criterions.rl import pick_action
from models.utils import entropy
from .utils import save_model, plot_accuracy_and_error


def train(setup,                            # setup dict, including model and training info
          agent,                            # agent to be trained
          envs,                             # list of environments, each may use a different output from the agent
          optimizer,                        # optimizer used for updating the agent's parameters
          scheduler,                        # learning rate scheduler
          criterion,                        # loss criterion used for reinforcement learning
          sl_criterion,                     # loss criterion used for supervised learning
          ax_criterion=None,                # auxiliary loss criterion
          
          # the following parameters can be any order, don't insert parameters between parameters above
          model_save_path=None,             # path to save the trained model
          device='cpu',                     # device
          use_memory=None,                  # whether to use memory module
          num_iter=10000,                   # number of training iterations
          test_iter=200,                    # number of iterations between testing (outputting results)
          save_iter=1000,                   # number of iterations between saving the model
          min_iter=0,                       # minimum number of iterations before stopping the training
          stop_test_accu=1.0,               # accuracy threshold to stop the training
          reset_memory=True,                # whether to reset the memory module before each trial
          used_output_index=[0],            # list of indices specifying the output used for each environment
          env_sample_prob=[1.0],            # list of probabilities for sampling each environment
          grad_clip=True,                   # whether to clip the gradients during optimization
          grad_max_norm=1.0,                # maximum norm value for gradient clipping
          session_num=1,                    # session number for saving the accuracy and error plot
          mem_beta_decay_threshold=None,    # performance threshold for decaying softmax beta
          mem_beta_decay_iter=10000,        # number of iterations between decaying softmax beta, 
                                                # if mem_beta_decay_threshold is not None, decay based on the performance
          sl_criterion_weight=1.0,          # weight for the supervised learning loss
    ):
    """
    Trains the agent using the specified environments and optimization parameters.

    Returns:
        test_accuracies (list): List of accuracies computed during testing.
        test_errors (list): List of errors computed during testing.
    """
    torch.autograd.set_detect_anomaly(True)
    # each used_output_index corresponds to an environment
    # specifying the output that is used for computing the action for stepping the environment
    assert len(envs) == len(used_output_index) == len(env_sample_prob)

    # set up some parameters and variables
    total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, total_actor_loss, \
        total_critic_loss, total_sl_loss, total_entropy = 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
    reward_masks = 0.0
    
    num_iter = int(num_iter)
    # test_accuracies = np.zeros(num_iter // test_iter + 5)
    # test_errors = np.zeros(num_iter // test_iter + 5)
    training_time = np.zeros(num_iter // test_iter + 5)
    test_accuracies = []
    test_errors = []
    test_rewards = []
    # training_time = []
    test_times = 0
    # test_accuracies, test_errors = [], []

    batch_size = envs[0].num_envs
    if use_memory:
        agent.use_memory = use_memory
    min_test_loss = torch.inf

    print("start training")
    print("batch size:", batch_size)

    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    
    loss = 0.0
    decay_mem_beta = False

    # compute time for each process
    env_time, forward_time, loss_time, backward_time, total_time, env_step_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(num_iter):
        # record time for the first iteration to estimate total time needed
        if i == 0:
            start_time = time.time()

        t_start = time.time()

        # randomly sample an environment from the list of environments
        env_id = np.random.choice(len(envs), p=env_sample_prob)
        env = envs[env_id]
        batch_size = env.num_envs

        # before each trial, for the agent:
        # 1. reset initial state
        # 2. reset memory module
        state = agent.init_state(batch_size, decay_mem_beta=decay_mem_beta)
        agent.reset_memory(flush=reset_memory)

        # create variables to store data related to outputs and results
        try:
            seq_len = setup["training"]["env"][0]["sequence_len"]
        except:
            seq_len = setup["training"]["env"][0]["tasks"][0]["sequence_len"]

        try:
            try:
                retrieve_time_limit = setup["training"]["env"][0]["retrieve_time_limit"]
            except:
                retrieve_time_limit = setup["training"]["env"][0]["tasks"][0]["retrieve_time_limit"]
        except:
            retrieve_time_limit = seq_len


        # rewards = np.zeros((seq_len*2, batch_size))
        rewards = []
        gts = np.zeros((seq_len+retrieve_time_limit, batch_size, len(agent.output_dims)))
        gt_masks = np.zeros((seq_len+retrieve_time_limit, batch_size), dtype=bool)
        loss_masks = np.zeros((seq_len+retrieve_time_limit, batch_size), dtype=bool)

        outputs = defaultdict(list)
        # actions, probs, actions_max = defaultdict(list), defaultdict(list), defaultdict(list)
        probs, actions_max = defaultdict(list), defaultdict(list)
        actions = []
        values, entropys = defaultdict(list), defaultdict(list)
        model_infos = defaultdict(list)
        mem_similarities, mem_sim_entropys = [], []
        # action_space_masks = []

        """ run the agent in the environment """
        correct_actions, wrong_actions, not_know_actions = 0, 0, 0

        # reset environment
        obs_, info = env.reset()
        obs = torch.Tensor(obs_).to(device)
        # print(env.memory_sequence)
        terminated = np.zeros(batch_size, dtype=bool)
        memory_num = 0
        timestep = 0
        while not terminated.all():
            # set up the phase of the agent
            if info["phase"][0] == "encoding":
                memory_num += 1
                agent.set_encoding(True)
                agent.set_retrieval(False)
            elif info["phase"][0] == "recall":
                agent.set_encoding(False)
                agent.set_retrieval(True)
            # reset state between phases
            if "reset_state" in info and info["reset_state"][0]:
                state = agent.init_state(batch_size, recall=True, prev_state=state)

            t1 = time.time()

            # do one step of forward pass for the agent
            # output: batch_size x action_space, value: batch_size x 1
            output, value, state, model_info = agent(obs, state)

            t2 = time.time()


            t7 = time.time()
            action, log_prob_action, action_max = pick_action(output)
            # obs_, reward, _, _, info = env.step(list(action))
            obs_, reward, _, _, info = env.step(action.cpu().detach().numpy().transpose(1, 0))
            if "reward" in info:
                reward = info["reward"]     # reward is a list of rewards for each task
            env_step_time += time.time() - t7
            done = info["done"]
            obs = torch.Tensor(obs_).to(device)
            # rewards[timestep] = reward
            if "sum_reward" in info and info["sum_reward"].all():
                rewards.append(np.sum(reward, axis=1))      # the reward used for RL is the sum, but calculate reward history separately
            else:
                rewards.append(reward)

            # print(gts.shape, info["gt"].shape)
            gts[timestep] = np.array(info["gt"])
            gt_masks[timestep] = info["gt_mask"]
            loss_masks[timestep] = np.logical_and(info["loss_mask"], np.logical_not(terminated))

            correct_actions += np.sum(info["correct"], axis=0)
            wrong_actions += np.sum(info["wrong"], axis=0)
            not_know_actions += np.sum(info["not_know"], axis=0)
            terminated = np.logical_or(terminated, done)

            for j, o in enumerate(output):
                outputs[j].append(o)
                probs[j].append(log_prob_action[j])
                # actions[j].append(action[j])
                actions_max[j].append(action_max[j])
                values[j].append(value[j])
                entropys[j].append(entropy(o, device))
            actions.append(action)

            for key in model_info.keys():
                model_infos[key].append(model_info[key])

            mem_similarities.append(model_info["memory_similarity"])
            mem_sim_entropys.append(entropy(model_info["memory_similarity"], device))
            # mem_similarities[timestep] = model_info["memory_similarity"]
            # mem_sim_entropys[timestep] = entropy(model_info["memory_similarity"], device)
            total_reward += np.sum(np.array(reward))   

            t3 = time.time()

            forward_time += t2 - t1
            env_time += t3 - t2

            timestep += 1

        """ compute the loss and do backpropagation """
        if (i+1) % test_iter == 0:
            print_criterion_info = True
            print('Action distribution:', output[0][0])
        else:
            print_criterion_info = False

        for j in range(len(outputs)):
            probs[j] = probs[j][memory_num:]
            values[j] = values[j][memory_num:]
            entropys[j] = entropys[j][memory_num:]
            outputs[j] = torch.stack(outputs[j]).to(device)
        # print(outputs[0].shape)

        for key in model_infos:
            model_infos[key] = torch.stack(model_infos[key]).to(device)

        t4 = time.time()

        # compute RL loss
        if criterion is not None:
            loss_rl, loss_actor, loss_critic, loss_ent_reg = criterion(probs, values, rewards[memory_num:], entropys, 
                                                                       loss_masks[memory_num:], print_info=print_criterion_info, 
                                                                       device=device)
            loss += loss_rl

            total_loss += loss_rl.item()
            total_actor_loss += loss_actor.item()
            total_critic_loss += loss_critic.item()
            t = torch.stack(entropys[used_output_index[env_id]]).cpu().detach().numpy().astype(np.float32)
            total_entropy += np.mean(t)

        # compute SL loss

        gts = torch.tensor(np.array(gts), dtype=torch.long).to(device)    # time x batch_size
        gt_masks = torch.tensor(np.array(gt_masks)).to(device)
        # print(gts.shape, gt_masks.shape, gts[gt_masks].shape)
        # print(gts, gt_masks)
        if sl_criterion is not None:
            # print(gts, gt_masks, outputs[0].shape)
            loss_sl = sl_criterion([outputs[o][gt_masks] for o in outputs], gts[gt_masks])
            loss += loss_sl * sl_criterion_weight
            total_sl_loss += loss_sl.item()

        # compute auxiliary loss
        if ax_criterion is not None:
            loss_ax = ax_criterion(device=device, **model_infos)
            loss += loss_ax
            if print_criterion_info:
                print("auxiliary loss:", loss_ax.item())

        t5 = time.time()

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_max_norm, error_if_nonfinite=True)
        optimizer.step()
        loss = 0.0

        t6 = time.time()

        loss_time += t5 - t4
        backward_time += t6 - t5

        # compute accuracy
        actions_total_num += correct_actions + wrong_actions + not_know_actions
        actions_correct_num += correct_actions
        actions_wrong_num += wrong_actions

        total_time += time.time() - t_start

        decay_mem_beta = False

        """ print training information and save model """
        if (i+1) % test_iter == 0:
            accuracy = np.round(actions_correct_num / (actions_total_num+1e-10), 2)
            error = np.round(actions_wrong_num / (actions_total_num+1e-10), 2)
            not_know_rate = np.round(1 - accuracy - error, 2)
            mean_loss = total_loss / (test_iter * batch_size)
            mean_actor_loss = total_actor_loss / (test_iter * batch_size)
            mean_critic_loss = total_critic_loss / (test_iter * batch_size)
            mean_entropy = total_entropy / (test_iter * batch_size)

            mean_reward = total_reward / (test_iter * batch_size)

            print('Iteration: {},  train accuracy: {}, error: {}, no action: {}, mean reward: {:2f}, total loss: {:.4f}, actor loss: {:.4f}, '
                'critic loss: {:.4f}, entropy: {:.4f}'.format(i+1, accuracy, error, not_know_rate, mean_reward, mean_loss, mean_actor_loss, mean_critic_loss,
                                                              mean_entropy))
            actions_trial = torch.stack(actions).cpu().detach().numpy().transpose(0, 2, 1)  # timesteps x batch_size x action_num
            gts_trial = gts.cpu().detach().numpy()  # timesteps x batch_size x action_num
            # print(actions_trial.shape, gts_trial.shape)
            if sl_criterion is not None:
                print("encoding phase, action:", actions_trial[0:memory_num, 0].reshape(-1), "gt:", gts_trial[0:memory_num, 0].reshape(-1))
            if criterion is not None:
                print("recall phase, action:", actions_trial[memory_num:, 0].reshape(-1), "gt:", gts_trial[memory_num:, 0].reshape(-1))

            # update learning rate
            if i != 0:
                scheduler.step(-mean_reward)  # TODO: change a criterion here?
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                if lr != current_lr:
                    print("lr changed from {} to {}".format(current_lr, lr))
                    current_lr = lr

            # decide whether to decay mem_beta based on decay iter and decay threshold of performance
            if (i+1) % mem_beta_decay_iter == 0:
                if mem_beta_decay_threshold is not None:
                    if np.max(accuracy) >= mem_beta_decay_threshold:
                        decay_mem_beta = True
                    else:
                        decay_mem_beta = False
                else:
                    decay_mem_beta = True

            # save model
            # if error - accuracy <= min_test_loss:
            #     min_test_loss = error - accuracy
            #     save_model(agent, model_save_path, filename="model.pt")
            save_model(agent, model_save_path, filename="model.pt")
            
            # stop training if accuracy is high enough
            if np.min(accuracy) >= stop_test_accu and i > min_iter:
                if agent.mem_beta_decay and agent.mem_beta <= agent.mem_beta_min or not agent.mem_beta_decay:
                    print("training end")
                    break

            # plot accuracy and error
            # test_accuracies[test_times] = accuracy
            # test_errors[test_times] = error
            test_accuracies.append(accuracy)
            test_errors.append(error)
            test_rewards.append(mean_reward)
            plot_accuracy_and_error(test_accuracies, test_errors, model_save_path, filename="accuracy_session_{}.png".format(session_num))
            np.save(model_save_path/"accuracy_{}.npy".format(session_num), np.array(test_accuracies))
            np.save(model_save_path/"error_{}.npy".format(session_num), np.array(test_errors))
            np.save(model_save_path/"reward_{}.npy".format(session_num), np.array(test_rewards))

            # reset training variables
            total_reward, actions_correct_num, actions_wrong_num, actions_total_num, total_loss, \
                total_actor_loss, total_critic_loss, total_entropy = 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0

            # print training time and estimated time needed
            print("total time: {:.2f}s, env time: {:.2f}s, env step time: {:.2f}s, forward time: {:.2f}s, loss time: {:.2f}s, backward time: {:.2f}s".format(
                total_time, env_time, env_step_time, forward_time, loss_time, backward_time))
            env_time, forward_time, loss_time, backward_time, total_time, env_step_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            # training_time[test_times] = time.time() - start_time
            # print("Estimated time needed: {:2f}h".format(np.mean(training_time[:test_times+1])/test_iter*(num_iter-test_iter*test_times)/3600))
            # start_time = time.time()

            training_time[test_times] = time.time() - start_time
            estimated_time_seconds = np.mean(training_time[:test_times+1]) / test_iter * (num_iter - test_iter * test_times)
            estimated_time = timedelta(seconds=estimated_time_seconds)
            print("Estimated time needed: {}".format(str(estimated_time)[:-3]))  # Remove microseconds
            start_time = time.time()

            print()
            test_times += 1
        
        if (i+1) % save_iter == 0:
            save_model(agent, model_save_path, filename="{}_{}.pt".format(session_num, i))
    
    return test_accuracies, test_errors
