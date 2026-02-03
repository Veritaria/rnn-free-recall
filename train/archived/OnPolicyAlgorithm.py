from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from models.utils import entropy, softmax
from .criterions.rl import pick_action
from .utils import save_model, plot_accuracy_and_error


@dataclass
class Rollout:
    # Episode length actually used (<= max_steps)
    # T: time, B: batch size, A: action space
    length: int
    memory_num: int
    batch_size: int
    n_action_spaces: int

    # Stored on device for fast replay (PPO) and losses
    obs: torch.Tensor  # (T, B, *obs_shape)

    # Stored on CPU (cheap + easy masking)
    phases: np.ndarray  # (T,) 0=encoding, 1=recall
    reset_state: np.ndarray  # (T,) bool

    rewards: np.ndarray  # (T, B) float32 (RL reward used for returns)
    loss_masks: np.ndarray  # (T, B) bool
    gt_masks: np.ndarray  # (T, B) bool
    action_space_masks: np.ndarray  # (T, A, B) bool
    gts: np.ndarray  # (T, B, A) int64

    # Per action-space tensors on device
    actions: torch.Tensor  # (T, A, B) long
    log_probs: torch.Tensor  # (T, A, B) float
    values: torch.Tensor  # (T, A, B, 1) float
    entropys: torch.Tensor  # (T, A, B) float

    # For SL + auxiliary losses (computed from the rollout forward pass)
    outputs: List[torch.Tensor]  # list[A] of (T, B, action_dim)
    model_infos: Dict[str, torch.Tensor]  # key -> (T, B, ...)


class OnPolicyAlgorithm:
    """
    Stable-baselines-style skeleton for on-policy RL.

    - `collect_rollout()` runs one full trial/episode in one env.
    - `train_on_rollouts()` applies the algorithm-specific update (A2C vs PPO).
    - supervised + auxiliary losses stay in the shared code path.
    """

    def __init__(
        self,
        setup: dict,
        agent,
        envs: Sequence,
        optimizer,
        scheduler,
        criterion,
        sl_criterion,
        ax_criterion=None,
        *,
        model_save_path=None,
        device: str = "cpu",
        use_memory: Optional[bool] = None,
        reset_memory: bool = True,
        used_output_index: Sequence[int] = (0,),
        env_sample_prob: Sequence[float] = (1.0,),
        grad_clip: bool = True,
        grad_max_norm: float = 1.0,
        sl_criterion_weight: float = 1.0,
        eta=None,
        epochs=1,
    ):
        assert len(envs) == len(used_output_index) == len(env_sample_prob)

        self.setup = setup
        self.agent = agent
        self.envs = list(envs)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.sl_criterion = sl_criterion
        self.ax_criterion = ax_criterion

        self.model_save_path = model_save_path
        self.device = device
        self.reset_memory = reset_memory
        self.used_output_index = list(used_output_index)
        self.env_sample_prob = np.array(env_sample_prob, dtype=np.float64)
        self.grad_clip = grad_clip
        self.grad_max_norm = grad_max_norm
        self.sl_criterion_weight = sl_criterion_weight
        self.eta = eta
        self.epochs = epochs

        if use_memory:
            self.agent.use_memory = use_memory

    def _get_max_steps(self) -> int:
        try:
            seq_len = self.setup["training"]["env"][0]["sequence_len"]
        except Exception:
            seq_len = self.setup["training"]["env"][0]["tasks"][0]["sequence_len"]
        return int(seq_len) * 2

    def collect_rollout(self, env, *, decay_mem_beta: bool = False) -> Tuple[Rollout, Dict[str, Any]]:
        """
        Run one complete environment trial and return a preallocated rollout.
        """
        agent = self.agent
        device = self.device

        # Reset env
        obs_np, info = env.reset()
        obs0 = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        batch_size = obs0.shape[0]

        # Reset agent
        state = agent.init_state(batch_size, decay_mem_beta=decay_mem_beta)
        agent.reset_memory(flush=self.reset_memory)

        n_action_spaces = len(agent.output_dims)
        max_steps = self._get_max_steps()

        # Preallocate buffers
        obs_buf = torch.empty((max_steps,) + tuple(obs0.shape), device=device, dtype=torch.float32)
        phases = np.zeros((max_steps,), dtype=np.int8)
        reset_state = np.zeros((max_steps,), dtype=bool)

        # rewards = np.zeros((max_steps, batch_size), dtype=np.float32)
        rewards = []
        loss_masks = np.zeros((max_steps, batch_size), dtype=bool)
        gt_masks = np.zeros((max_steps, batch_size), dtype=bool)
        action_space_masks = np.zeros((max_steps, n_action_spaces, batch_size), dtype=bool)
        gts = np.zeros((max_steps, batch_size, n_action_spaces), dtype=np.int64)

        actions = torch.empty((max_steps, n_action_spaces, batch_size), device=device, dtype=torch.long)
        log_probs = torch.empty((max_steps, n_action_spaces, batch_size), device=device, dtype=torch.float32)
        values = torch.empty((max_steps, n_action_spaces, batch_size, 1), device=device, dtype=torch.float32)
        entropys = torch.empty((max_steps, n_action_spaces, batch_size), device=device, dtype=torch.float32)

        outputs = [
            torch.empty((max_steps, batch_size, int(agent.output_dims[i])), device=device, dtype=torch.float32)
            for i in range(n_action_spaces)
        ]

        model_info_lists: Dict[str, List[torch.Tensor]] = {}

        terminated = np.zeros(batch_size, dtype=bool)
        memory_num = 0
        t = 0

        # Episode stats
        correct_actions = 0
        wrong_actions = 0
        not_know_actions = 0
        total_reward = 0.0
        last_reward = None
        last_info = info

        obs = obs0
        while not terminated.all():
            if t > max_steps:
                # Safety: avoid infinite loops if env config mismatch
                break

            phase_str = last_info["phase"][0]
            if phase_str == "encoding":
                phases[t] = 0
                memory_num += 1
                agent.set_encoding(True)
                agent.set_retrieval(False)
            else:
                phases[t] = 1
                agent.set_encoding(False)
                agent.set_retrieval(True)

            if "reset_state" in last_info and last_info["reset_state"][0]:
                reset_state[t] = True
                state = agent.init_state(batch_size, recall=True, prev_state=state)

            obs_buf[t].copy_(obs)

            output_list, value_list, state, model_info = agent(obs, state)

            # action distribution per action space
            action_dists = [softmax(o, beta=agent.softmax_beta) for o in output_list]
            action_as, log_prob_as, _ = pick_action(action_dists)  # (A,B)

            # step env
            obs_next_np, reward, _, _, info_next = env.step(action_as.detach().cpu().numpy().transpose(1, 0))
            if "reward" in info_next:
                reward = np.stack(np.array(info_next["reward"]), axis=0)  # (B, n_task) or (B,) depending on env

            if "sum_reward" in info_next and info_next["sum_reward"].all():
                # sum reward across all tasks
                reward_rl = np.sum(reward, axis=1)
            else:
                reward_rl = reward
            total_reward += np.sum(reward, axis=0)
            last_reward = reward

            # RL reward used by loss should be (B,)
            # if isinstance(reward, np.ndarray) and reward.ndim == 2:
            #     reward_rl = reward.sum(axis=0)
            #     total_reward += reward.sum(axis=0)
            #     last_reward = reward
            # else:
            #     reward_rl = np.array(reward, dtype=np.float32)
            #     total_reward += float(np.sum(reward_rl))
            #     last_reward = reward_rl

            # masks + labels from env info
            gts[t] = np.stack(info_next["gt"], axis=0)  # (B, A)
            gt_masks[t] = np.array(info_next["gt_mask"], dtype=bool)
            loss_masks[t] = np.logical_and(np.array(info_next["loss_mask"], dtype=bool), np.logical_not(terminated))
            action_space_masks[t] = np.stack(info_next["action_space_mask"], axis=0).T.astype(bool)  # (A,B)

            # accuracy stats
            correct_actions += np.sum(np.stack(info_next["correct"], axis=0), axis=0)
            wrong_actions += np.sum(np.stack(info_next["wrong"], axis=0), axis=0)
            not_know_actions += np.sum(np.stack(info_next["not_know"], axis=0), axis=0)

            terminated = np.logical_or(terminated, np.array(info_next["done"], dtype=bool))

            rewards.append(reward_rl.astype(np.float32))
            actions[t] = action_as
            log_probs[t] = log_prob_as
            for a_i in range(n_action_spaces):
                outputs[a_i][t].copy_(output_list[a_i])
                values[t, a_i].copy_(value_list[a_i])
                entropys[t, a_i].copy_(entropy(action_dists[a_i], device))

            # model infos for aux losses
            for k, v in model_info.items():
                model_info_lists.setdefault(k, []).append(v)

            obs = torch.as_tensor(obs_next_np, dtype=torch.float32, device=device)
            last_info = info_next
            t += 1

        rewards = np.stack(rewards, axis=0)

        # Stack model infos (T, B, ...)
        model_infos = {k: torch.stack(vs[:t]).to(device) for k, vs in model_info_lists.items() if len(vs) > 0}

        rollout = Rollout(
            length=t,
            memory_num=memory_num,
            batch_size=batch_size,
            n_action_spaces=n_action_spaces,
            obs=obs_buf[:t],
            phases=phases[:t],
            reset_state=reset_state[:t],
            rewards=rewards,
            loss_masks=loss_masks[:t],
            gt_masks=gt_masks[:t],
            action_space_masks=action_space_masks[:t],
            gts=gts[:t],
            actions=actions[:t],
            log_probs=log_probs[:t],
            values=values[:t],
            entropys=entropys[:t],
            outputs=[o[:t] for o in outputs],
            model_infos=model_infos,
        )

        episode_info = {
            "correct_actions": correct_actions,
            "wrong_actions": wrong_actions,
            "not_know_actions": not_know_actions,
            "total_reward": total_reward,
            "last_reward": last_reward,
            "last_info": last_info,
        }
        return rollout, episode_info

    def _sl_loss(self, rollout: Rollout, outputs: List[torch.Tensor]) -> torch.Tensor:
        if self.sl_criterion is None:
            return torch.tensor(0.0, device=self.device)

        T, B, A = rollout.gts.shape
        gt_masks = torch.as_tensor(rollout.gt_masks[:T], device=self.device)
        gts = torch.as_tensor(rollout.gts[:T], dtype=torch.long, device=self.device)

        flat_mask = gt_masks.reshape(-1)
        gts_flat = gts.reshape(-1, A)

        selected_outputs = []
        for a_i, out in enumerate(outputs):
            out_flat = out.reshape(-1, out.shape[-1])
            selected_outputs.append(out_flat[flat_mask])

        return self.sl_criterion(selected_outputs, gts_flat[flat_mask], device=self.device)

    def _aux_loss(self, model_infos: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.ax_criterion is None or not model_infos:
            return torch.tensor(0.0, device=self.device)
        return self.ax_criterion(device=self.device, **model_infos)

    def train_on_rollouts(self, rollouts: List[Rollout], *, print_info: bool = False) -> Dict[str, float]:
        """
        Algorithm-specific update. Must run backward/step internally and return a stats dict.
        """
        raise NotImplementedError

    def learn(
        self,
        *,
        num_iter: int = 10000,
        test_iter: int = 200,
        save_iter: int = 1000,
        min_iter: int = 0,
        stop_test_accu: float = 1.0,
        session_num: int = 1,
        mem_beta_decay_threshold=None,
        mem_beta_decay_iter: int = 10000,
        num_rollouts: int = 1,
    ):
        num_iter = int(num_iter)

        total_reward = 0.0
        actions_correct_num = 0
        actions_wrong_num = 0
        actions_total_num = 0
        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_sl_loss = 0.0
        total_entropy = 0.0
        reward_masks: Any = 0.0

        training_time = np.zeros(num_iter // test_iter + 5)
        test_accuracies: List[Any] = []
        test_errors: List[Any] = []
        test_rewards: List[Any] = []
        test_times = 0

        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        decay_mem_beta = False

        rollouts: List[Rollout] = []
        current_env_id: Optional[int] = None

        # time for each process
        total_time = 0.0
        rollout_time = 0.0
        train_time = 0.0

        print("start training")
        print("batch size:", self.envs[0].num_envs)

        t0 = time.time()

        for i in range(num_iter):
            if i == 0:
                start_time = time.time()

            t1 = time.time()

            # For algorithms that batch multiple rollouts (PPO), keep env consistent within the batch.
            if current_env_id is None:
                current_env_id = int(np.random.choice(len(self.envs), p=self.env_sample_prob))
            env = self.envs[current_env_id]

            rollout, ep_info = self.collect_rollout(env, decay_mem_beta=decay_mem_beta)
            rollouts.append(rollout)
            decay_mem_beta = False  # reset decay_mem_beta to False after each rollout

            t2 = time.time()
            rollout_time += t2 - t1

            # reward mask is for computing the mean reward reported when testing
            # reward mask records how many trials there are for each task
            last_reward = ep_info["last_reward"]
            last_info = ep_info["last_info"]
            if isinstance(last_reward, np.ndarray) and last_reward.ndim == 1:
                reward_masks += rollout.batch_size
            else:
                if "reward_mask" in last_info:
                    reward_masks += np.sum(last_info["reward_mask"], axis=0)
                else:
                    # default: one mask per task
                    if isinstance(last_reward, np.ndarray) and last_reward.ndim == 2:
                        reward_masks += np.ones(last_reward.shape[0]) * rollout.batch_size
                    else:
                        reward_masks += rollout.batch_size

            total_reward += ep_info["total_reward"]
            actions_total_num += ep_info["correct_actions"] + ep_info["wrong_actions"] + ep_info["not_know_actions"]
            actions_correct_num += ep_info["correct_actions"]
            actions_wrong_num += ep_info["wrong_actions"]

            # train only when we have enough rollouts collected
            if len(rollouts) < int(num_rollouts):
                continue

            t3 = time.time()

            print_criterion_info = (i + 1) % test_iter == 0
            stats = self.train_on_rollouts(rollouts, print_info=print_criterion_info)

            t4 = time.time()
            train_time += t4 - t3

            # aggregate stats (count only once per update)
            total_loss += float(stats.get("loss", 0.0))
            total_actor_loss += float(stats.get("actor_loss", 0.0))
            total_critic_loss += float(stats.get("critic_loss", 0.0))
            total_entropy += float(stats.get("entropy", 0.0))
            total_sl_loss += float(stats.get("sl_loss", 0.0))

            rollouts = []
            current_env_id = None

            decay_mem_beta = False

            total_time += time.time() - t0
            t0 = time.time()


            if (i + 1) % test_iter == 0:
                accuracy = np.round(actions_correct_num / (actions_total_num + 1e-10), 2)
                error = np.round(actions_wrong_num / (actions_total_num + 1e-10), 2)
                not_know_rate = np.round(1 - accuracy - error, 2)

                mean_loss = total_loss / (test_iter * rollout.batch_size)
                mean_actor_loss = total_actor_loss / (test_iter * rollout.batch_size)
                mean_critic_loss = total_critic_loss / (test_iter * rollout.batch_size)
                mean_entropy = total_entropy / (test_iter * rollout.batch_size)
                mean_sl_loss = total_sl_loss / (test_iter * rollout.batch_size)

                if not isinstance(reward_masks, int):
                    reward_masks = np.array(reward_masks)
                mean_reward = np.round(total_reward / np.array(reward_masks).reshape(-1, 1), 3)

                print(
                    'Iteration: {},  train accuracy: {}, error: {}, no action: {}, mean reward: {}, total loss: {:.4f}, actor loss: {:.4f}, '
                    'critic loss: {:.4f}, entropy: {:.4f}'.format(
                        i + 1, accuracy, error, not_know_rate, mean_reward, mean_loss, mean_actor_loss, mean_critic_loss, mean_entropy
                    )
                )

                if self.sl_criterion is not None:
                    print("sl loss: {:.4f}".format(mean_sl_loss))

                print("total time: {:.2f}s, rollout time: {:.2f}s, train time: {:.2f}s".format(total_time, rollout_time, train_time))
                total_time, rollout_time, train_time = 0.0, 0.0, 0.0

                if i != 0:
                    self.scheduler.step(-np.sum(mean_reward))
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    if lr != current_lr:
                        print("lr changed from {} to {}".format(current_lr, lr))
                        current_lr = lr

                if (i + 1) % mem_beta_decay_iter == 0:
                    if mem_beta_decay_threshold is not None:
                        decay_mem_beta = bool(np.max(accuracy) >= mem_beta_decay_threshold)
                    else:
                        decay_mem_beta = True

                save_model(self.agent, self.model_save_path, filename="model.pt")

                if np.min(accuracy) >= stop_test_accu and i > min_iter:
                    print("training end")
                    break

                test_accuracies.append(accuracy)
                test_errors.append(error)
                test_rewards.append(mean_reward)
                plot_accuracy_and_error(
                    test_accuracies, test_errors, self.model_save_path, filename="accuracy_session_{}.png".format(session_num)
                )
                np.save(self.model_save_path / "accuracy_{}.npy".format(session_num), np.array(test_accuracies))
                np.save(self.model_save_path / "error_{}.npy".format(session_num), np.array(test_errors))
                np.save(self.model_save_path / "reward_{}.npy".format(session_num), np.array(test_rewards))

                total_reward = 0.0
                actions_correct_num = 0
                actions_wrong_num = 0
                actions_total_num = 0
                total_loss = 0.0
                total_actor_loss = 0.0
                total_critic_loss = 0.0
                total_sl_loss = 0.0
                total_entropy = 0.0
                reward_masks = 0.0

                training_time[test_times] = time.time() - start_time
                estimated_time_seconds = np.mean(training_time[: test_times + 1]) / test_iter * (num_iter - test_iter * test_times)
                estimated_time = timedelta(seconds=estimated_time_seconds)
                print("Estimated time needed: {}".format(str(estimated_time)[:-3]))
                start_time = time.time()

                print()
                test_times += 1

            if (i + 1) % save_iter == 0:
                save_model(self.agent, self.model_save_path, filename="{}.pt".format(i))

        return test_accuracies, test_errors

