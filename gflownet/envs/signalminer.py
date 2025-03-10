"""
SignalMiner environment: starting from an emtpy sequence, hyper-params are added one by one up
to a maximum length.
"""

from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tlong

import numpy as np

PARAM_DICT = {
    'colsample_bytree': list(np.linspace(0.01, 1, 10)), 
    'reg_lambda': list(np.linspace(0, 100_000, 10)),
    'learning_rate': list( np.linspace(.00001, 1.0, 10, dtype='float') ),
    'max_bin' : list(np.linspace(2, 5, 4, dtype='int')),
    'max_depth': list(np.linspace(2, 16, 10, dtype='int')),# [5, 10, 15, 20, 25, 50, 100],
    'num_leaves': list(np.linspace(2, 32, 10, dtype='int')),#, 4112],#, 8192, 32768],
    'min_child_samples': list( np.linspace(1,1000,10,dtype='int') ),
    'n_estimators': list( np.linspace(1,100,10,dtype='int') ),#,75,100,150,200],#, 500, 700, 900, 1200], 
    # 'boltzmann_alpha':list( np.linspace(-30,30,61,dtype='int') ),
    # 'target':targets,
    # 'num_train_eras': list( np.linspace(4,64,60,dtype='int') ),
    # 'split_type': ['directional_erasplit', 'original']
    # 'split_type': ['directional_erasplit']
}

class SignalMiner(GFlowNetEnv):
    """
    SignalMiner environment: sequences are constructed starting from an empty sequence and
    adding one hyper-parameter at a time.

    States are represented by a list of indices corresponding to each hyper-parameter, starting
    from 1, and are padded with index 0.

    Actions are represented by a single-element tuple with the index of the hyper-parameter to
    be added. The EOS action is by (-1, ).

    Attributes
    ----------
     param_dict : dict
        A dict containing the hyper-parameters to use. By default, PARAM_DICT is used.

    pad_token : str
       PAD token. Default: "0".
    """

    def __init__(
        self,
        param_dict: dict = None,
        pad_token: str = "0",
        **kwargs,
    ):
        # Main attributes
        if param_dict is None:
            self.param_dict = PARAM_DICT
        else:
            self.param_dict = param_dict

        self.param_names = []
        self.param_lengths = []
        
        for k in self.param_dict.keys():
            self.param_names.append(k)
            self.param_lengths.append(len(PARAM_DICT[k]))
        
        self.pad_token = pad_token
        # self.n_letters = len(self.letters)
        self.max_length = len(self.param_dict)
        self.max_param_length = max(self.param_lengths)
        self.eos_idx = -1
        self.pad_idx = 0
        # Dictionaries
        # self.idx2token = {idx + 1: token for idx, token in enumerate(self.letters)}
        # self.idx2token[self.pad_idx] = pad_token
        # self.token2idx = {token: idx for idx, token in self.idx2token.items()}
        # Source state: list of length max_length filled with pad token
        self.source = [self.pad_idx] * self.max_length
        # End-of-sequence action
        self.eos = (self.eos_idx,)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including eos.

        An action is represented by a single-element tuple indicating the index of the
        letter to be added to the current sequence (state).

        The action space of this parent class is:
            action_space: [(0,), (1,), (-1,)]
        """
        return [(n,) for n in range(1, self.max_param_length + 1)] + [(self.eos_idx,)]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid (should be masked).
            - False otherwise (action can be selected).
    
        Args
        ----
        state : list[int]
            Input state. If None, self.state is used.
    
        done : bool
            Whether the trajectory is done. If None, self.done is used.
    
        Returns
        -------
        A list of boolean values indicating which actions are invalid.
        """
        state = self._get_state(state)
        done = self._get_done(done)
    
        if done:
            return [True] * self.action_space_dim  # Mask all actions if done
    
        # Initialize all actions as valid
        this_action_space = [False] * self.action_space_dim
    
        # Find the current step (which hyperparameter is being chosen)
        step = self._get_seq_length(state)
    
        # Get the number of available choices for the current hyperparameter
        if step < self.max_length:
            step_len = self.param_lengths[step]  # Number of valid choices for this step
        else:
            step_len = 0  # If we are done, no choices left
    
        # Mask out indices greater than the available options
        for i in range(step_len, self.max_param_length):
            this_action_space[self.action_space.index((i + 1,))] = True  # Mark as invalid
    
        # Ensure EOS is ONLY valid at the final step
        eos_idx = self.action_space.index(self.eos)
        if step < self.max_length:  
            this_action_space[eos_idx] = True  # Mask EOS if not at the last step
        else:
            this_action_space = [True] * self.action_space_dim  # Mask everything
            this_action_space[eos_idx] = False  # Allow EOS
    
        return this_action_space



    def get_parents(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.
    
        The GFlowNet graph is a tree and there is only one parent per state.
    
        Args
        ----
        state : list[int]
            Input state. If None, self.state is used.
    
        done : bool
            Whether the trajectory is done. If None, self.done is used.
    
        action : None
            Ignored.
    
        Returns
        -------
        parents : list
            List of parents in state format. This environment has a single parent per state.
    
        actions : list
            List of actions that lead to state for each parent in parents. This
            environment has a single parent per state.
        """
        state = self._get_state(state)
        done = self._get_done(done)
    
        # If the state is a fully defined hyperparameter set, its only parent is itself
        if done:
            return [state], [self.eos]
    
        # If at the root state (empty hyperparameter set), there are no parents
        if self.equal(state, self.source):
            return [], []
    
        # Find the last hyperparameter that was selected
        last_selected_idx = self._get_seq_length(state) - 1
    
        # Create the parent state by resetting the last selected hyperparameter to "unfilled"
        parent = copy(state)
        parent[last_selected_idx] = self.pad_idx
    
        # The action that led from the parent state to the current state
        previous_action = (state[last_selected_idx],)
    
        return [parent], [previous_action]


    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> [List[int], Tuple[int], bool]:
        """
        Executes a step by selecting a hyperparameter value.
    
        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple containing an integer index that
            selects a hyperparameter value.
    
        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.
    
        Returns
        -------
        self.state : list
            The sequence after executing the action.
    
        action : tuple
            The action that was executed.
    
        valid : bool
            False if the action is not allowed for the current state.
        """
        # Generic pre-step checks (ensures action is valid)
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False  # Invalid action, return unchanged state
    
        valid = True
        self.n_actions += 1
    
        # If action is EOS, mark the sequence as complete
        if action == self.eos:
            self.done = True
            return self.state, action, valid
    
        # Determine the current hyperparameter index to fill
        step = self._get_seq_length(self.state)
    
        # Update state with the selected hyperparameter value
        self.state[step] = action[0]
    
        return self.state, action, valid


    def states2proxy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: the batch is
        simply converted into a tensor of indices.
    
        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.
    
        Returns
        -------
        A list containing all the states in the batch, represented as lists.
        """
        return tlong(states, device=self.device)
    
    
    def states2policy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model.
    
        - Each hyperparameter index is one-hot encoded.
        - If a hyperparameter hasn't been chosen yet (pad_idx), it's encoded as all zeros.
    
        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.
    
        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tlong(states, device=self.device)  # Convert to long tensor
    
        # One-hot encode the hyperparameter values
        one_hot_states = [
            F.one_hot(states[:, i], num_classes=self.param_lengths[i] + 1)  # 🔥 Custom encoding per param
            for i in range(self.max_length)
        ]
    
        # Concatenate all one-hot encodings into a single feature vector per state
        return torch.cat(one_hot_states, dim=1).to(self.float)


    def state2readable(self, state: List[int] = None) -> str:
        """
        Converts a state into a human-readable string.
    
        - Instead of letters, this will now display hyperparameter names & chosen values.
        - Unselected (pad_idx) parameters will be shown as 'None'.
    
        Args
        ----
        state : list[int]
            A state in environment format. If None, self.state is used.
    
        Returns
        -------
        A formatted string displaying hyperparameter values.
        """
        state = self._get_state(state)
        state = self._unpad(state)  # Remove padding
    
        readable_str = []
        for i, param_value in enumerate(state):
            param_name = self.param_names[i]  # Get hyperparameter name
            if param_value == self.pad_idx:
                readable_str.append(f"{param_name}: None")  # Unfilled hyperparameters
            else:
                readable_str.append(f"{param_name}: {self.param_dict[param_name][param_value-1]}")
    
        return " | ".join(readable_str)  # Format as readable output
    
    
    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a readable hyperparameter string back into a state.
    
        Args
        ----
        readable : str
            A formatted string like 'learning_rate: 0.1 | max_depth: 10 | ...'
    
        Returns
        -------
        A tensor containing the indices of the selected hyperparameter values.
        """
        if readable.strip() == "":
            return self.source  # Return initial state if empty
    
        state = []
        for param_str in readable.split(" | "):
            param_name, param_value = param_str.split(": ")
            param_value = param_value.strip()
    
            if param_value == "None":
                state.append(self.pad_idx)  # Unselected parameter
            else:
                # Convert value back to index
                param_idx = self.param_dict[param_name].index(float(param_value)) + 1
                state.append(param_idx)
    
        return self._pad(state)  # Ensure correct length


    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List[int]]:
        """
        Constructs a batch of `n_states` uniformly sampled from the hyperparameter space.
    
        Each state is a list of indices, with one index per hyperparameter.
    
        Args
        ----
        n_states : int
            The number of states to sample.
    
        seed : int
            Random seed for reproducibility.
    
        Returns
        -------
        A list of `n_states` complete hyperparameter configurations, sampled uniformly.
        """
        if seed is not None:
            torch.manual_seed(seed)
    
        # Initialize empty state tensor
        samples = torch.zeros((n_states, self.max_length), dtype=torch.long, device=self.device)
    
        # Sample a random valid index for each hyperparameter
        for i in range(self.max_length):
            param_size = self.param_lengths[i]  # Number of valid choices for this hyperparameter
            samples[:, i] = torch.randint(0, param_size, (n_states,), device=self.device)  # Random selection
    
        return samples.tolist()


    def _pad(self, param_list: list):
        """
        Pads a sequence of hyperparameter values with `pad_idx` until `max_length`.
    
        Args
        ----
        param_list : list
            A list containing selected hyperparameter values.
    
        Returns
        -------
        The input list padded with `self.pad_idx` up to `self.max_length`.
        """
        return param_list + [self.pad_idx] * (self.max_length - len(param_list))
    
    
    def _unpad(self, param_list: list):
        """
        Removes padding from a sequence of hyperparameter values.
    
        Args
        ----
        param_list : list
            A list containing hyperparameter indices, possibly with padding.
    
        Returns
        -------
        The input list with `pad_idx` values removed.
        """
        if self.pad_idx not in param_list:
            return param_list  # No padding, return as is
        return param_list[: param_list.index(self.pad_idx)]  # Trim at first `pad_idx`
    
    
    def _get_seq_length(self, state: List[int] = None):
        """
        Returns the number of hyperparameters that have been selected so far.
    
        Args
        ----
        state : list
            A list representing a hyperparameter selection state. If None, uses `self.state`.
    
        Returns
        -------
        The number of hyperparameters chosen (ignoring padding).
        """
        state = self._get_state(state)
    
        if self.pad_idx in state:
            return state.index(self.pad_idx)  # First unfilled hyperparameter
        else:
            return len(state)  # Fully filled state
