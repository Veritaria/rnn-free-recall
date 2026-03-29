import math
import torch
import torch.nn as nn
from copy import deepcopy

from ..utils import load_act_fn, softmax
from ..base_module import BasicModule
from ..memory import ValueMemory
from ..module.encoders import MLPEncoder
from ..module.decoders import ActorCriticMLPDecoder


class ValueMemoryGRU(BasicModule):
    def __init__(self, 
                 memory_module: ValueMemory, 
                 hidden_dim: int, 
                 input_dim: int, 
                 output_dims: list, 
                 input_encoder_dims: list = [],
                 output_encoder_dims: list = [],
                 em_gate_type='constant',
                 init_state_type="zeros", 
                 evolve_state_between_phases=False, 
                 evolve_steps=1, 
                 softmax_beta=1.0, 
                 wm_noise_prop=0, 
                 em_noise_prop=0, 
                 wm_enc_noise_prop=0,
                 wm_em_zero_noise=False,              # if true, make the noise for wm/em to be all zero (only cancel the original data but do not add noise)
                 start_recall_with_ith_item_init=0, 
                 reset_param=True,
                 step_for_each_timestep=1, 
                 flush_noise=0.1, 
                 random_init_noise=0.1, 
                 layer_norm=False, 
                 use_memory=True, 
                 mem_beta_decay=False,             # whether to decay softmax beta for computing memory similarity loss
                 mem_beta_decay_rate=0.5,          # decay rate for softmax beta
                 mem_beta_min=1e-6,                # minimum value for softmax beta
                 reservoir=False,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.memory_module = memory_module      # memory module of the model, pre-instantiated

        self.use_memory = use_memory            # if false, do not use memory module in the forward pass

        self.step_for_each_timestep = step_for_each_timestep

        # encoding and retrieval status, if true, do memory encoding or/and retrieval in the forward pass
        self.encoding = False
        self.retrieving = False

        self.wm_noise_prop = wm_noise_prop      # noise proportion for working memory
        self.em_noise_prop = em_noise_prop      # noise proportion for episodic memory
        self.wm_enc_noise_prop = wm_enc_noise_prop   # noise proportion for working memory only during encoding phase
        self.wm_em_zero_noise = 1 - float(wm_em_zero_noise)      # if true, make the noise for wm/em to be all zero (only cancel the original data but do not add noise)

        self.softmax_beta = softmax_beta        # 1/temperature for softmax function for computing final output decision
        if mem_beta_decay:
            self.mem_beta = torch.nn.Parameter(torch.tensor(self.memory_module.similarity_measure.softmax_temperature, dtype=torch.float32), requires_grad=False)
            print("mem_beta initialized to {}".format(self.mem_beta.item()))
        else:
            self.mem_beta = None
        self.mem_beta_decay = mem_beta_decay
        self.mem_beta_decay_rate = mem_beta_decay_rate
        self.mem_beta_min = mem_beta_min
        
        self.flush_noise = flush_noise
        self.random_init_noise = random_init_noise
        self.layer_norm = layer_norm

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        if isinstance(output_dims, int):
            output_dims = [output_dims]
        self.output_dims = output_dims          # there could be mutliple output decisions

        # fc_hidden_dim = int(hidden_dim/4)

        self.encoder = MLPEncoder(input_dim, 3 * hidden_dim, hidden_dims=input_encoder_dims)
        self.fc_hidden = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.decoders = nn.ModuleList()
        for output_dim in output_dims:
            self.decoders.append(ActorCriticMLPDecoder(hidden_dim, output_dim, hidden_dims=output_encoder_dims))

        self.ln_i2h = torch.nn.LayerNorm(2*hidden_dim, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2*hidden_dim, elementwise_affine=False)
        self.ln_cell_1 = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # gate when adding episodic memory to hidden state
        self.em_gate_type = em_gate_type
        if em_gate_type == "constant":
            self.em_gate = 1.0
        elif em_gate_type == "scalar" or em_gate_type == "scalar_sigmoid":
            self.em_gate = nn.Linear(hidden_dim, 1)
        elif em_gate_type == "vector":
            self.em_gate = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Invalid em_gate_type: {em_gate_type}")

        # if true, compute forward pass for an extra timestep between encoding and retrieval phases
        self.evolve_state_between_phases = evolve_state_between_phases
        self.evolve_steps = evolve_steps
        self.last_encoding = False

        self.start_recall_with_ith_item_init = start_recall_with_ith_item_init
        self.ith_item_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)
        self.current_timestep = 0

        self.init_state_type = init_state_type
        if init_state_type == "train":
            # train initial hidden state
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        if self.init_state_type == 'train_diff':
            # train different initial hidden state for encoding and recall phase
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
            self.h0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

        if reset_param:
            self.reset_parameters()
        else:
            self.reset_parameters2()

        if reservoir:
            for name, param in self.named_parameters():
                if 'hidden' in name or 'ln' in name:
                    param.requires_grad = False

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            if w.requires_grad:
                w.data.uniform_(-std, std)

    def reset_parameters2(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            if w.requires_grad:
                w.data.normal_(0.0, std)

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None, decay_mem_beta=False):
        if recall:
            # initialize hidden state for recall phase
            if self.start_recall_with_ith_item_init != 0:
                state = self.ith_item_state.clone()
            elif self.init_state_type == 'zeros':
                state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == 'noise' or self.init_state_type == 'noise_all':
                # state = math.sqrt(1 - self.flush_noise) * prev_state + math.sqrt(self.flush_noise) * torch.randn_like(prev_state) * torch.std(prev_state)
                state = math.sqrt(1 - self.flush_noise**2) * prev_state + self.flush_noise * torch.randn_like(prev_state) * torch.std(prev_state)
            elif self.init_state_type == 'noise_random' or self.init_state_type == 'noise_random_all':
                # noise, but random the flush_noise
                flush_noise = self.flush_noise[0] + (self.flush_noise[1] - self.flush_noise[0]) * torch.rand((batch_size, 1), device=self.device)
                # state = torch.sqrt(1 - flush_noise) * prev_state + torch.sqrt(flush_noise) * torch.randn_like(prev_state) * torch.std(prev_state)
                # state = torch.sqrt(1 - flush_noise**2) * prev_state + flush_noise * torch.randn_like(prev_state) * torch.std(prev_state)
                state = (1 - flush_noise) * prev_state + torch.sqrt(1-(1-flush_noise)**2) * torch.randn_like(prev_state) * torch.std(prev_state)
            elif self.init_state_type == 'random':
                # state = torch.randn((batch_size, self.hidden_dim), device=self.device, requires_grad=True) * self.random_init_noise
                state = self.random_init_state.clone()
            elif self.init_state_type == 'train':
                state = self.h0.repeat(batch_size, 1)
            elif self.init_state_type == 'train_diff':
                state = self.h0_recall.repeat(batch_size, 1)
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, noise, noise_all, noise_random, noise_random_all, random, train or train_diff")
            state = torch.tanh(state)
        else:
            # initialize hidden state for encoding phase
            if self.init_state_type == "zeros" or self.init_state_type == "noise" or self.init_state_type == "noise_random":
                state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == "train" or self.init_state_type == "train_diff":
                state = torch.tanh(self.h0.repeat(batch_size, 1))
            elif self.init_state_type == "random" or self.init_state_type == "noise_all" or self.init_state_type == "noise_random_all":
                self.random_init_state = torch.randn((batch_size, self.hidden_dim), device=self.device, requires_grad=True) * self.random_init_noise
                state = self.random_init_state.clone()
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, noise, noise_all, noise_random, noise_random_all, random, train or train_diff")
        
        if self.mem_beta_decay and decay_mem_beta and self.mem_beta > self.mem_beta_min:
            self.mem_beta = torch.nn.Parameter(self.mem_beta * self.mem_beta_decay_rate, requires_grad=False)
            print("mem_beta decayed to {}".format(self.mem_beta.item()))
        
        self.write(state, 'init_state')
        return state

    def forward(self, inp, state, beta=None):
        batch_size = inp.shape[0]

        if self.last_encoding and self.evolve_state_between_phases and self.retrieving:
            for _ in range(self.evolve_steps):
                inp0 = torch.zeros_like(inp)
                # do a timestep of forward pass between encoding and retrieval phases
                state = self.gru(inp0, state)
            self.last_encoding = False
            self.write(state, 'state')

        # retrieve memory
        if self.use_memory and self.retrieving:
            # mem_beta = self.mem_beta if mem_beta is None else mem_beta
            retrieved_memory, memory_similarity = self.memory_module.retrieve(state, beta=self.mem_beta.item() if self.mem_beta is not None else None)
            if self.em_gate_type == "constant":
                mem_gate = self.em_gate
            elif self.em_gate_type == "scalar_sigmoid" or self.em_gate_type == "vector":
                mem_gate = self.em_gate(state).sigmoid()
            elif self.em_gate_type == "scalar":
                mem_gate = self.em_gate(state)
            else:
                raise ValueError(f"Invalid em_gate_type: {self.em_gate_type}")
            self.write(mem_gate, 'mem_gate_recall')
            self.write(memory_similarity, 'memory_similarity')
            self.write(retrieved_memory, 'retrieved_memory')
        else:
            retrieved_memory = torch.zeros(batch_size, self.hidden_dim)
            mem_gate = 0.0
            memory_similarity = torch.zeros(batch_size, self.memory_module.capacity)

        if self.use_memory and self.encoding:
            state = math.sqrt(1 - self.wm_enc_noise_prop) * state + math.sqrt(self.wm_enc_noise_prop) * torch.randn_like(state) * torch.std(state) * self.wm_em_zero_noise

        # compute forward pass
        for i in range(self.step_for_each_timestep):
            state = self.gru(inp, state, mem_gate, retrieved_memory)
        
        # store memory
        if self.use_memory and self.encoding:
            self.memory_module.encode(state)
            self.last_encoding = True
            self.current_timestep += 1
            if self.current_timestep == self.start_recall_with_ith_item_init:
                self.ith_item_state = state.detach().clone()

        self.write(state, 'state')

        beta = self.softmax_beta if beta is None else beta
        decisions, values = [], []
        for i in range(len(self.decoders)):
            decision, value = self.decoders[i](state)
            decision = softmax(decision, beta)
            decisions.append(decision)
            values.append(value)
            self.write(decision, 'decision{}'.format(i+1))
            self.write(value, 'value{}'.format(i+1))

        self.write(self.use_memory, 'use_memory')
        info = {
            "memory_similarity": memory_similarity
        }
        
        return decisions, values, state, info
    
    def gru(self, inp, state, mem_gate=None, retrieved_memory=None):
        gate_x = self.encoder(inp)
        gate_h = self.fc_hidden(state)
        if self.layer_norm:
            i_r, i_i = self.ln_i2h(gate_x[:, :2*self.hidden_dim]).chunk(2, 1)
            h_r, h_i = self.ln_h2h(gate_h[:, :2*self.hidden_dim]).chunk(2, 1)
            i_n = self.ln_cell_1(gate_x[:, 2*self.hidden_dim:])
            h_n = self.ln_cell_2(gate_h[:, 2*self.hidden_dim:])
        else:
            i_r, i_i, i_n = gate_x.chunk(3, 1)
            h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate_preact = i_n + resetgate * h_n
        if mem_gate is not None and retrieved_memory is not None:
            retrieved_memory = math.sqrt(1 - self.em_noise_prop) * retrieved_memory + \
                math.sqrt(self.em_noise_prop) * torch.randn_like(retrieved_memory) * torch.std(retrieved_memory) * self.wm_em_zero_noise
            newgate_preact += mem_gate * retrieved_memory
        newgate = torch.tanh(newgate_preact)
        state = newgate + inputgate * (state - newgate)
        state = math.sqrt(1 - self.wm_noise_prop) * state + math.sqrt(self.wm_noise_prop) * torch.randn_like(state) * torch.std(state) * self.wm_em_zero_noise
        return state

    def set_encoding(self, status):
        """
        set memory encoding status of the model
        """
        self.encoding = status
        self.memory_module.encoding = status
    
    def set_retrieval(self, status):
        """
        set memory retrieval status of the model
        """
        self.retrieving = status
        self.memory_module.retrieving = status

    def reset_memory(self, flush=True):
        """
        reset memory of the memory module of the model
        """
        self.memory_module.reset_memory(flush=flush)
