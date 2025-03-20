from typing import List, Union

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

import lightgbm as lgb

from numerapi import NumerAPI
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class SignalMinerProxy(Proxy):
    """
    Proxy function that assigns a reward to each hyperparameter set
    based on its performance in training.
    """

    def __init__(self, **kwargs):
        """
        Initializes the proxy.
        """
        super().__init__(**kwargs)
        self.env = None
        self.param_dict = None
        self.param_names = None

        napi = NumerAPI()
        napi.download_dataset("v5.0/train.parquet", "train.parquet")

        data = pd.read_parquet("train.parquet")

        targets = [t for t in data.columns if 'target' in t]
        feature_cols = [c for c in data.columns if 'feature' in c][:50]
        
        data['era'] = data['era'].astype('int')
        data[targets] = (data[targets] * 4).astype('Int8')
        
        eras = np.array(sorted(data['era'].unique()))
        all_splits = list(TimeSeriesSplit(n_splits=20, max_train_size=100_000_000, gap=4).split(eras))

        train_didxs, test_didxs = all_splits[0]

        self.data = data
        self.target = targets[0]
        self.feature_cols = feature_cols
        self.train_didxs = train_didxs
        self.test_didxs = test_didxs
        self.eras = eras

        label = self.target#cfg['target']
        train_rows = (self.data['era'].isin(self.eras[self.train_didxs])) & (~self.data[label].isna())
        test_rows = (self.data['era'].isin(self.eras[self.test_didxs])) & (~self.data[label].isna())

        self.train_data = self.data.loc[train_rows, self.feature_cols].values
        self.train_target = self.data.loc[train_rows, label].values

        self.test_data = self.data.loc[ test_rows, self.feature_cols].values
        self.test_rows = test_rows
    

    def setup(self, env):
        """
        Set up the proxy with an environment, ensuring compatibility with GFlowNet.

        Args
        ----
        env : SignalMiner
            The environment containing hyperparameter configurations.
        """
        self.env = env
        self.param_names = self.env.param_names
        self.param_dict = self.env.param_dict

    def __call__(
        self, states: Union[List[List[int]], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        """
        Computes the reward (Sharpe ratio) for each hyperparameter configuration.

        Args
        ----
        states : tensor or list
            A batch of states where each row represents a set of hyperparameter indices.

        Returns
        -------
        A tensor of reward scores (Sharpe ratios).
        """
        if torch.is_tensor(states):
            states = states.tolist()  # Convert tensor to list of lists

        scores = []
        for state in states:
            # Convert state from index representation to hyperparameter values
            hp_config = self._convert_state_to_hyperparams(state)

            # Compute model performance (Sharpe ratio)
            sharpe_ratio = self._evaluate_hyperparams(hp_config)

            # Store the result
            scores.append(sharpe_ratio)

        return tfloat(scores, device=self.device, float_type=self.float)

    def _convert_state_to_hyperparams(self, state: List[int]) -> dict:
        """
        Converts a state vector into a hyperparameter dictionary.

        Args
        ----
        state : list
            A list of hyperparameter indices.

        Returns
        -------
        A dictionary with hyperparameter names and their selected values.
        """
        if self.param_names is None or self.param_dict is None:
            raise ValueError("Proxy has not been set up with an environment.")

        hp_config = {}
        for i, idx in enumerate(state):
            param_name = self.param_names[i]
            hp_config[param_name] = self.param_dict[param_name][idx-1]

        return hp_config

    def _evaluate_hyperparams(self, hp_config: dict) -> float:
        """
        Evaluates a hyperparameter configuration and returns its Sharpe ratio.

        Args
        ----
        hp_config : dict
            A dictionary of hyperparameters.

        Returns
        -------
        A float representing the Sharpe ratio.
        """
        try:
            # Train the model with these hyperparameters and return Sharpe ratio

    
            model = self.get_model(hp_config)
            
    
            model.fit( self.train_data, self.train_target )

            
            self.data.loc[ self.test_rows, 'pred'] = model.predict(self.test_data)
            validation_era_results = self.data.loc[ self.test_rows ].groupby('era')[[self.target, 'pred']].apply(lambda x: x[[self.target, 'pred']].dropna().corr().iloc[0, 1]).values
            sharpe_ratio = np.exp( np.nanmean(validation_era_results) * 100 )# / np.nanstd(validation_era_results) )
        except Exception as e:
            print(f"Error in evaluation: {e}")
            sharpe_ratio = 0.0000001  # Assign worst reward in case of failure

        return sharpe_ratio

    def softplus(self, x, beta=1.0):
        return (1 / beta) * np.log1p(np.exp(beta * x))
        
    def get_model(self, cfg):
        model = lgb.LGBMRegressor(
            n_jobs=2,
            verbosity=-1
        )

        # print(cfg)
    
        # Ensure cfg is a dictionary or iterable of key-value pairs
        model.set_params(**dict(cfg))  # Convert cfg to dict and unpack
    
        return model
