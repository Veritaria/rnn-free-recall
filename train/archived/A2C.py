from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .OnPolicyAlgorithm import OnPolicyAlgorithm, Rollout


class A2C(OnPolicyAlgorithm):
    """
    Algorithm-specific update for A2C.
    Everything else (rollout collection, SL + auxiliary losses, logging) is shared in `OnPolicyAlgorithm`.
    """

    def train_on_rollouts(self, rollouts: List[Rollout], *, print_info: bool = False) -> Dict[str, float]:
        assert len(rollouts) == 1, "A2C expects a single rollout per update."
        r = rollouts[0]

        t0 = int(r.memory_num)
        T = int(r.length)
        A = int(r.n_action_spaces)

        # RL tensors (recall only)
        probs = {}
        values = {}
        entropys = {}
        for a_i in range(A):
            probs[a_i] = list(r.log_probs[t0:T, a_i].unbind(0))
            values[a_i] = list(r.values[t0:T, a_i].unbind(0))
            entropys[a_i] = list(r.entropys[t0:T, a_i].unbind(0))

        rewards = r.rewards[t0:T]  # (T, B)
        loss_masks = r.loss_masks[t0:T]  # (T, B)
        action_space_masks = r.action_space_masks[t0:T]  # (T, A, B)

        if self.criterion is not None:
            loss_rl, loss_actor, loss_critic, loss_ent = self.criterion(
                probs,
                values,
                rewards,
                entropys,
                loss_masks,
                print_info=print_info,
                action_space_masks=action_space_masks,
                eta=self.eta,
                device=self.device,
                old_log_probs=None
            )
            # loss = loss + loss_rl
            actor_loss = float(loss_actor.detach().cpu().item())
            critic_loss = float(loss_critic.detach().cpu().item())
            ent_term = float(loss_ent.detach().cpu().item())
        else:
            loss_rl = torch.tensor(0.0, device=self.device)
            actor_loss = torch.tensor(0.0, device=self.device)
            critic_loss = torch.tensor(0.0, device=self.device)
            ent_term = torch.tensor(0.0, device=self.device)

        loss_sl = self._sl_loss(r, r.outputs)
        loss_ax = self._aux_loss(r.model_infos)
        loss = loss_rl + loss_sl * self.sl_criterion_weight + loss_ax

        # entropy stat (match older reporting: mean entropy of a selected output head)
        used_head = int(self.used_output_index[0]) if len(self.used_output_index) > 0 else 0
        entropy_mean = float(torch.mean(r.entropys[t0:T, used_head]).detach().cpu().item()) if T > t0 else 0.0

        # backward/step (single update)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_max_norm, error_if_nonfinite=True)
        self.optimizer.step()

        stats: Dict[str, float] = {
            "loss": float(loss.detach().cpu().item()),
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_mean,
            "sl_loss": float(loss_sl.detach().cpu().item()) if self.sl_criterion is not None else 0.0,
        }
        return stats

