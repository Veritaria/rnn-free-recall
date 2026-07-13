import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseEMTask


class ConditionalFreeRecall(BaseEMTask):
    def __init__(self, fix_one_feature=False, no_action_reward=-0.0, fix_condition=False, **kwargs):
        super().__init__(no_action_reward=no_action_reward, **kwargs)
        self.fix_one_feature = fix_one_feature
        self.fix_condition = fix_condition

    def reset(self, memory_sequence_index=None, **kwargs):
        obs, info = super().reset(memory_sequence_index=memory_sequence_index, **kwargs)
        # self._generate_task_condition()
        self.retrieved = np.zeros(self.vocabulary_size, dtype=bool)

        self.no_action_num = 0
        if self.fix_one_feature:
            fixed_feature = np.random.choice(self.num_features)
            all_stimuli = self.all_stimuli[self.all_stimuli[:, fixed_feature] == self.condition_value]
            self.memory_sequence = all_stimuli[np.random.choice(len(all_stimuli), self.sequence_len, replace=False)]
            if self.fix_condition:
                self.condition_feature = fixed_feature
                self.matched_item_num = 0
                for i in range(self.sequence_len):
                    if self.memory_sequence[i, self.condition_feature] == self.condition_value:
                        self.matched_item_num += 1
            self.memory_sequence_int = self._convert_item_to_int(self.memory_sequence)
            obs = self._generate_observation(self.memory_sequence[0], None, self.condition_feature, self.condition_value, 
                                         include_condition=self.include_condition_during_encode)
        
        self.matched_item_indexes = np.where(self.memory_sequence[:, self.condition_feature] == self.condition_value)[0]

        return obs, info

    
    def step(self, action):
        obs, reward, done, _, info = super().step(action)
        return obs, reward, done, False, info
    

    def compute_accuracy(self, actions):
        corrects, wrongs, not_knows = 0, 0, 0

        retrieved = np.ones(self.vocabulary_size, dtype=bool)
        for action in actions:
            action_item = self._convert_action_to_item(action)
            action_item_int = self._convert_item_to_int(action_item)
            if action_item_int < 0:
                not_knows += 1
            elif action_item_int in self.memory_sequence_int[self.matched_item_indexes] and not retrieved[action_item_int]:
                corrects += 1
                retrieved[action_item_int] = True
            else:
                wrongs += 1
        return corrects, wrongs, not_knows

    
    def get_trial_data(self):
        """
        used when recording data
        """
        return {
            "memory_sequence": self.memory_sequence,
            "condition_feature": self.condition_feature,
            "condition_value": self.condition_value,
            "memory_sequence_int": self.memory_sequence_int + 1,
            "matched_item_indexes": self.matched_item_indexes
        }


    def _generate_condition_features(self):
        iter_num = 0
        while True:
            self.condition_feature = np.random.choice(self.num_features)
            self.condition_value = np.random.choice(self.feature_dim)
            if np.sum(self.memory_sequence[:, self.condition_feature] == self.condition_value) > 0:
                break
        
        self.matched_item_num = 0
        for i in range(self.sequence_len):
            if self.memory_sequence[i, self.condition_feature] == self.condition_value:
                self.matched_item_num += 1

    
    def _compute_reward_and_metrics(self, action):
        action_item = self._convert_action_to_item(action)
        action_item_int = self._convert_item_to_int(action_item.reshape(1, -1))
        # print("action_item_int", action_item_int)
        correct, wrong, not_know = 0, 0, 0
        if action_item_int in self.memory_sequence_int[self.matched_item_indexes] \
            and not self.retrieved[action_item_int]:
            correct += 1
            reward = self.correct_reward
            self.retrieved[action_item_int] = True
        elif action_item_int < 0:
            not_know += 1
            reward = self.no_action_reward
            self.no_action_num += 1
        else:
            wrong += 1
            reward = self.wrong_reward

        # if last time step, give a penalty of all not recalled items 
        if self.timestep == self.sequence_len:
            reward += self.wrong_reward * min((self.matched_item_num - np.sum(self.retrieved)), self.no_action_num)

        return reward, correct, wrong, not_know


    def _compute_gt(self):
        gt = np.zeros(len(self.action_space.nvec))
        ind = min(self.timestep-1, self.sequence_len-1)
        if self.one_hot_action:
            if (self.phase == "recall" or self.include_condition_during_encode) and\
                self.memory_sequence_int[self.timestep-1] not in self.memory_sequence_int[self.matched_item_indexes]: 
                gt[0] = 0
            else:
                item_int = self._convert_item_to_int(self.memory_sequence[self.timestep-1].reshape(1, -1)) + 1
                gt[0] = item_int[0]
        else:
            if (self.phase == "recall" or self.include_condition_during_encode) and\
                self.memory_sequence_int[self.timestep-1] not in self.memory_sequence_int[self.matched_item_indexes]:
                # item doesn't match condition
                gt[-1] = 1
            else:
                gt[:self.num_features] = self.memory_sequence[self.timestep-1]
                gt[-1] = 0
        return gt
        