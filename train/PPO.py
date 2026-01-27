from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from models.utils import softmax, entropy
from .OnPolicyAlgorithm import OnPolicyAlgorithm, Rollout


class PPO(OnPolicyAlgorithm):
    """
    PPO update with replay of stored trajectories (to compute new log-probs under the current policy).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _replay_rollouts(
        self, rollout: Rollout
    ) -> Tuple[
        Dict[int, List[torch.Tensor]],
        Dict[int, List[torch.Tensor]],
        Dict[int, List[torch.Tensor]],
        List[torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """
        Replay a rollout through the *current* policy to compute:
        - new log_probs (recall steps only)
        - values/entropys (recall steps only)
        - outputs (all steps, for supervised loss)
        - model_infos (all steps, for auxiliary loss)
        """
        device = self.device
        agent = self.agent

        r = rollout

        # Accumulators
        n_action_spaces = r.n_action_spaces
        probs: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_action_spaces)}
        values: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_action_spaces)}
        entropys: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_action_spaces)}

        outputs_all: List[List[torch.Tensor]] = [[] for _ in range(n_action_spaces)]
        model_info_lists: Dict[str, List[torch.Tensor]] = {}

        # reset memory and state for this rollout
        B = r.batch_size
        state = agent.init_state(B, decay_mem_beta=False)
        agent.reset_memory(flush=self.reset_memory)

        T = r.length
        t0 = r.memory_num

        for t in range(T):
            # phase control (matches original rollout)
            if r.phases[t] == 0:
                agent.set_encoding(True)
                agent.set_retrieval(False)
            else:
                agent.set_encoding(False)
                agent.set_retrieval(True)

            if r.reset_state[t]:
                state = agent.init_state(B, recall=True, prev_state=state)

            out_list, val_list, state, model_info = agent(r.obs[t], state)

            # store outputs for SL
            for a_i in range(n_action_spaces):
                outputs_all[a_i].append(out_list[a_i])

            # store model infos for aux
            for k, v in model_info.items():
                model_info_lists.setdefault(k, []).append(v)

            # only recall steps contribute to PPO RL objective
            if t >= t0:
                action_dists = [softmax(o, beta=agent.softmax_beta) for o in out_list]
                for a_i in range(n_action_spaces):
                    m = torch.distributions.Categorical(action_dists[a_i])
                    act = r.actions[t, a_i]  # (B,)
                    probs[a_i].append(m.log_prob(act))
                    values[a_i].append(val_list[a_i])
                    entropys[a_i].append(entropy(action_dists[a_i], device))

        # stack outputs + model_infos (all steps)
        outputs_concat = [torch.stack(outputs_all[a_i]).to(device) for a_i in range(n_action_spaces)]
        model_infos = {k: torch.stack(vs).to(device) for k, vs in model_info_lists.items() if len(vs) > 0}

        return probs, values, entropys, outputs_concat, model_infos


    def train_on_rollouts(self, rollouts: List[Rollout], *, print_info: bool = False) -> Dict[str, float]:
        n_action_spaces = rollouts[0].n_action_spaces

        # Stats: report mean over "batches" (rollouts) for the first epoch only (for logging).
        stats_sum_first_epoch = {
            "loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "sl_loss": 0.0,
        }
        num_batches = max(1, len(rollouts))

        for epoch in range(self.epochs):
            for b_idx, r in enumerate(rollouts):
                t0 = int(r.memory_num)
                T = int(r.length)

                # old_log_probs/rewards/masks for this rollout (recall only)
                old_log_probs: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_action_spaces)}
                for a_i in range(n_action_spaces):
                    old_log_probs[a_i] = list(r.log_probs[t0:T, a_i].detach().unbind(0))

                rewards = r.rewards[t0:T]  # (T_recall, B)
                loss_masks = r.loss_masks[t0:T]  # (T_recall, B)
                action_space_masks = r.action_space_masks[t0:T]  # (T_recall, A, B)

                probs, values, entropys, outputs_concat, model_infos = self._replay_rollouts(r)

                loss_total = torch.tensor(0.0, device=self.device)
                actor_loss_val = 0.0
                critic_loss_val = 0.0

                if self.criterion is not None:
                    loss_rl, loss_actor, loss_critic, _loss_ent = self.criterion(
                        probs,
                        values,
                        rewards,
                        entropys,
                        loss_masks,
                        print_info=print_info and epoch == 0 and b_idx == 0,
                        action_space_masks=action_space_masks,
                        eta=self.eta,
                        device=self.device,
                        old_log_probs=old_log_probs,
                    )
                    loss_total = loss_total + loss_rl
                    actor_loss_val = float(loss_actor.detach().cpu().item())
                    critic_loss_val = float(loss_critic.detach().cpu().item())

                # supervised + auxiliary losses (computed on replayed forward pass)
                loss_sl = self._sl_loss(r, outputs_concat)
                loss_ax = self._aux_loss(model_infos)
                loss_total = loss_total + loss_sl * self.sl_criterion_weight + loss_ax

                if epoch == 0:
                    used_head = int(self.used_output_index[0]) if len(self.used_output_index) > 0 else 0
                    ent_mean = 0.0
                    if used_head in entropys and len(entropys[used_head]) > 0:
                        ent_mean = float(torch.mean(torch.stack(entropys[used_head])).detach().cpu().item())

                    stats_sum_first_epoch["loss"] += float(loss_total.detach().cpu().item())
                    stats_sum_first_epoch["actor_loss"] += float(actor_loss_val)
                    stats_sum_first_epoch["critic_loss"] += float(critic_loss_val)
                    stats_sum_first_epoch["entropy"] += float(ent_mean)
                    stats_sum_first_epoch["sl_loss"] += float(loss_sl.detach().cpu().item()) if self.sl_criterion is not None else 0.0

                # SGD step per *batch* (rollout) instead of per epoch
                self.optimizer.zero_grad()
                loss_total.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_max_norm, error_if_nonfinite=True)
                self.optimizer.step()

        stats_first_epoch = {k: (v / num_batches) for k, v in stats_sum_first_epoch.items()}
        return stats_first_epoch

