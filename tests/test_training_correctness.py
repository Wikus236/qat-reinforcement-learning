"""Check reproducible initialization and time-limit bootstrapping in all learners."""
import random
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch

from rl_models.dqn import DQN
from rl_models.qat import DQN_QAT
from rl_models.qat4 import DQN_QAT_4
from rl_models.qat_weights import DQN_QAT_W
from rl_models.qat_weights4 import DQN_QAT_W_4

LEARNERS=(DQN,DQN_QAT,DQN_QAT_4,DQN_QAT_W,DQN_QAT_W_4)


class TransitionEnv:
    """A continuing step, an external timeout, then a true terminal step."""
    observation_space=gym.spaces.Box(-1,1,shape=(8,),dtype=np.float32)
    action_space=gym.spaces.Discrete(4)

    def __init__(self):
        self.steps=0
        self.resets=0

    def reset(self,seed=None):
        self.resets+=1
        return np.zeros(8,dtype=np.float32),{}

    def step(self,action):
        self.steps+=1
        return (np.zeros(8,dtype=np.float32),float(self.steps),
                self.steps==3,self.steps==2,{})


def make(cls,seed=2):
    return cls('MlpPolicy',TransitionEnv(),policy_kwargs={'net_arch':[8,8]},
               seed=seed,learning_starts=1000,batch_size=3,gamma=.9,verbose=0)


class TrainingCorrectnessTests(unittest.TestCase):
    def test_initialization_is_seeded_before_both_networks(self):
        for cls in LEARNERS:
            with self.subTest(learner=cls.__name__):
                torch.manual_seed(123)
                first=make(cls,seed=2)
                torch.manual_seed(999)
                np.random.seed(999)
                random.seed(999)
                second=make(cls,seed=2)
                different=make(cls,seed=3)
                a=first.q_net.state_dict()
                self.assertTrue(all(torch.equal(v,second.q_net.state_dict()[k]) for k,v in a.items()))
                self.assertTrue(all(torch.equal(v,first.target_net.state_dict()[k]) for k,v in a.items()))
                weight=next(k for k,v in a.items() if k.endswith('weight') and v.ndim==2)
                self.assertFalse(torch.equal(a[weight],different.q_net.state_dict()[weight]))

    def test_rollouts_reset_at_timeouts_but_replay_masks_only_termination(self):
        for cls in LEARNERS:
            with self.subTest(learner=cls.__name__):
                model=make(cls)
                model.learn(total_timesteps=3)
                self.assertEqual(model.env.resets,3)
                self.assertEqual([bool(r[-1]) for r in model.replay_buffer.buffer],
                                 [False,False,True])

    def test_actual_training_targets_bootstrap_timeout_and_not_terminal(self):
        for cls in LEARNERS:
            with self.subTest(learner=cls.__name__):
                model=make(cls)
                model.learn(total_timesteps=3)
                model.q_net=torch.nn.Linear(8,4).to(model.device)
                model.target_net=torch.nn.Linear(8,4).to(model.device)
                with torch.no_grad():
                    model.q_net.weight.zero_();model.q_net.bias.zero_()
                    model.target_net.weight.zero_();model.target_net.bias.fill_(2.)
                model.optimizer=torch.optim.Adam(model.q_net.parameters(),lr=1e-4)
                # Targets are 1+.9*2, 2+.9*2 and 3. Q=0 gives mean Huber loss 2.7.
                self.assertAlmostEqual(model._train_step(),2.7,places=5)

    def test_saved_policy_predictions_and_weights_survive_load(self):
        for cls in LEARNERS:
            with self.subTest(learner=cls.__name__),tempfile.TemporaryDirectory() as tmp:
                model=make(cls)
                obs=np.linspace(-.3,.3,8,dtype=np.float32)
                action=model.predict(obs)[0]
                path=Path(tmp)/'policy.zip'
                model.save(path)
                loaded=cls.load(path,env=TransitionEnv())
                self.assertEqual(action,loaded.predict(obs)[0])
                self.assertTrue(all(torch.equal(v,loaded.q_net.state_dict()[k])
                                    for k,v in model.q_net.state_dict().items()))

    def test_real_lunarlander_rollout_and_gradient_updates(self):
        for cls in LEARNERS:
            with self.subTest(learner=cls.__name__):
                env=gym.make('LunarLander-v3',continuous=False,enable_wind=False)
                try:
                    env.action_space.seed(2)
                    model=cls('MlpPolicy',env,policy_kwargs={'net_arch':[8,8]},seed=2,
                              learning_starts=8,batch_size=8,train_freq=4,
                              target_update_interval=16,exploration_decay_steps=500000)
                    model.learn(total_timesteps=64)
                    self.assertEqual(len(model.replay_buffer),64)
                    self.assertTrue(model.optimizer.state)
                    self.assertTrue(all(torch.isfinite(p).all() for p in model.q_net.parameters()))
                finally:
                    env.close()


if __name__=='__main__':
    unittest.main()
