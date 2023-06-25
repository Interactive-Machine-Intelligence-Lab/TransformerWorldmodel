import random
import sys
from typing import List, Optional, Union

from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm
import wandb

from agent import Agent
from dataset import EpisodesDataset
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from utils import EpisodeDirManager, RandomHeuristic

class Collector:
    def __init__(self, env: Union[SingleProcessEnv, MultiProcessEnv], dataset: List[EpisodesDataset], episode_dir_manager: List[EpisodeDirManager]) -> None:
        self.env = env
        self.datasets = dataset
        self.episode_dir_managers = episode_dir_manager
        self.obs = self.env.reset()
        self.episode_ids = {} 
        self.heuristic = RandomHeuristic(self.env.num_actions)

    @torch.no_grad()
    def collect(self, agents: List[Agent], epoch: int, epsilon: float, should_sample: bool, temperature: float, burn_in: int, *, num_steps: Optional[int] = None, num_episodes: Optional[int] = None):
        assert all([self.env.num_actions == agent.world_model.act_vocab_size for agent in agents])
        assert 0 <= epsilon <= 1
        assert (num_steps is None) != (num_episodes is None)
        assert agents[0].device == agents[1].device
        
        should_stop = lambda steps, episodes: steps >= num_steps if num_steps is not None else episodes >= num_episodes

        to_log = []
        steps, episodes = 0, 0
        returns = []
        observations, actions, rewards, dones = {}, {}, {}, {}
        for agent_id in range(len(agents)):
            observations[agent_id] = []
            actions[agent_id] = []
            rewards[agent_id] = []
            dones[agent_id] = []
            self.episode_ids[agent_id] = [None] * self.env.num_envs
            

        burnin_obs_rec, mask_padding = None, None

        for agent_id, agent in enumerate(agents):
            if set(self.episode_ids[agent_id]) != {None} and burn_in > 0:
                current_episodes = [self.datasets[agent_id].get_episode(episode_id) for episode_id in self.episode_ids[agent_id]]
                segmented_episodes = [episode.segment(start=len(episode) - burn_in, stop=len(episode), should_pad=True) for episode in current_episodes]
                mask_padding = torch.stack([episode.mask_padding for episode in segmented_episodes], dim=0).to(agent.device)
                burnin_obs = torch.stack([episode.observations for episode in segmented_episodes], dim=0).float().div(255).to(agent.device)
                burnin_obs_rec = torch.clamp(agent.tokenizer.encode_decode(burnin_obs, should_preprocess=True, should_postprocess=True), 0, 1)

            agent.actor_critic.reset(n=self.env.num_envs, burnin_observations=burnin_obs_rec, mask_padding=mask_padding)
        pbar = tqdm(total=num_steps if num_steps is not None else num_episodes, desc=f'Experience collection ({self.datasets[0].name})', file=sys.stdout)

        while not should_stop(steps, episodes):

            acts = {}
            postfix = {}
            for agent_id, agent in enumerate(agents):
                observations[agent_id].append(self.obs[agent_id])
                obs = rearrange(torch.FloatTensor(self.obs[agent_id]).div(255), 'n h w c -> n c h w').to(agents[agent_id].device)
                act = agent.act(obs, should_sample=should_sample, temperature=temperature).cpu().numpy()
                if random.random() < epsilon:
                    act = self.heuristic.act(obs).cpu().numpy()
                acts[agent_id] = act[0]
                postfix[f'agent{agent_id}'] = act[0]
            
            pbar.set_postfix(postfix)
            self.obs, reward, done, _ = self.env.step(acts)

            for agent_id in range(len(agents)):
                rewards[agent_id].append(reward[agent_id])
                actions[agent_id].append([acts[agent_id]])
                dones[agent_id].append(done[agent_id])

            new_steps = len(self.env.mask_new_dones)
            steps += new_steps
            pbar.update(new_steps if num_steps is not None else 0)


            if self.env.should_reset():
                self.add_experience_to_dataset(observations, actions, rewards, dones, agent_num=len(agents))

                new_episodes = self.env.num_envs
                episodes += new_episodes
                pbar.update(new_episodes if num_episodes is not None else 0)
                for agent_id in range(len(agents)):
                    for episode_id in self.episode_ids[agent_id]:
                        episode = self.datasets[agent_id].get_episode(episode_id)
                        self.episode_dir_managers[agent_id].save(episode, episode_id, epoch)
                        metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
                        metrics_episode['episode_num'] = episode_id
                        metrics_episode['action_histogram'] = wandb.Histogram(np_histogram=np.histogram(episode.actions.numpy(), bins=np.arange(0, self.env.num_actions + 1) - 0.5, density=True))
                        to_log.append({f'{self.dataset.name}/{k}/agent{agent_id}': v for k, v in metrics_episode.items()})
                        returns.append(metrics_episode['episode_return'])

                self.obs = self.env.reset()
                self.episode_ids = {}
                observations, actions, rewards, dones = {}, {}, {}, {}

                for agent_id, agent in enumerate(agents):
                    agent.actor_critic.reset(n=self.env.num_envs)
                    observations[agent_id] = []
                    actions[agent_id] = []
                    rewards[agent_id] = []
                    dones[agent_id] = []
                    self.episode_ids[agent_id] = [None] * self.env.num_envs
                    
        # Add incomplete episodes to dataset, and complete them later.
        if len(observations[0]) > 0:
            self.add_experience_to_dataset(observations, actions, rewards, dones, agent_num=len(agents))

        for agent_id, agent in enumerate(agents):
            agent.actor_critic.clear()

            metrics_collect = {
                '#episodes': len(self.datasets[agent_id]),
                '#steps': sum(map(len, self.datasets[agent_id].episodes)),
            }
            if len(returns) > 0:
                metrics_collect['return'] = np.mean(returns)
            metrics_collect = {f'{self.datasets[agent_id].name}/{k}/agent{agent_id}': v for k, v in metrics_collect.items()}
            to_log.append(metrics_collect)

        return to_log

    def add_experience_to_dataset(self, observations: List[np.ndarray], actions: List[np.ndarray], rewards: List[np.ndarray], dones: List[np.ndarray], agent_num) -> None:
        assert len(observations) == len(actions) == len(rewards) == len(dones)
        for agent_id in range(agent_num):
            for i, (o, a, r, d) in enumerate(zip(*map(lambda arr: np.swapaxes(arr, 0, 1), [observations[agent_id], actions[agent_id], rewards[agent_id], dones[agent_id]]))):  # Make everything (N, T, ...) instead of (T, N, ...)
                episode = Episode(
                    observations=torch.ByteTensor(o).permute(0, 3, 1, 2).contiguous(),  # channel-first
                    actions=torch.LongTensor(a),
                    rewards=torch.FloatTensor(r),
                    ends=torch.LongTensor(d),
                    mask_padding=torch.ones(d.shape[0], dtype=torch.bool),
                )
                if self.episode_ids[agent_id][i] is None:
                    self.episode_ids[agent_id][i] = self.datasets[agent_id].add_episode(episode)
                else:
                    self.datasets[agent_id].update_episode(self.episode_ids[i], episode)
