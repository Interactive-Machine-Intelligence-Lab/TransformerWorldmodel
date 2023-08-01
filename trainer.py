from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from agent import Agent
from collector import Collector
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch
from models.actor_critic import ActorCritic
from models.world_model import WorldModel, TransformerConfig
from utils import configure_optimizer, EpisodeDirManager, set_seed, clean_state_dict

from config import *
from envs import make_unity_gym
from dataset import EpisodesDatasetRamMonitoring, EpisodesDataset

from models.tokenizer import Tokenizer, Encoder, Decoder, EncoderDecoderConfig
import datetime

def create_foldername():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    second = str(now.second).zfill(2)
    foldername = f"{year}-{month}-{day}_{hour}-{minute}-{second}"
    return foldername


class Trainer:
    def __init__(self):

        wandb.init(
            config=dict({
                "train_cfg" : train_cfg,
                "world_model_cfg" : worldmodel_cfg,
                "tokenizer_cfg" : tok_cfg,
                "env_cfg" : env_cfg,
                "collector_cfg" : col_cfg    
            }),
            reinit=True,
            resume=True,
            **train_cfg.wandb
        )

        if train_cfg.common.seed is not None:
            set_seed(train_cfg.common.seed)

        self.start_epoch = 1
        self.device = torch.device(train_cfg.common.device)
        self.cfg = train_cfg

        self.agent_num = env_cfg.agent_num
        self.agents = []
        self.optimizers_tokenizer = []
        self.optimizers_world_model = []
        self.optimizers_actor_critic = []

        self.base_output = Path('output') / create_foldername() if not train_cfg.common.resume else Path(train_cfg.common.resume_path)
        self.ckpt_dir = self.base_output / 'checkpoints'
        self.media_dir = self.base_output / 'media'
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'


        if not train_cfg.common.resume:
            self.base_output.mkdir(exist_ok=True, parents=True)
            self.ckpt_dir.mkdir(exist_ok=True, parents=False)
            self.media_dir.mkdir(exist_ok=True, parents=False)
            self.episode_dir.mkdir(exist_ok=True, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=True, parents=False)

            for agent_id in range(self.agent_num):
                agent_path = self.ckpt_dir / f'agent{agent_id}'
                agent_path.mkdir(parents=True, exist_ok=True)
                recon_path = self.reconstructions_dir / f'agent{agent_id}'
                recon_path.mkdir(parents=True, exist_ok=True)

        
        episode_managers_train = []
        episode_managers_test = []
        self.episode_managers_imagination = []

        for agent_id in range(self.agent_num):
            episode_manager_train = EpisodeDirManager(self.episode_dir / 'train' / f'agent{agent_id}', max_num_episodes=train_cfg.collector_train.num_episodes_to_save)
            episode_manager_test = EpisodeDirManager(self.episode_dir / 'test' / f'agent{agent_id}', max_num_episodes=train_cfg.collector_test.num_episodes_to_save)
            episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination' / f'agent{agent_id}', max_num_episodes=train_cfg.evaluation_settings.actor_critic.num_episodes_to_save)
            episode_managers_train.append(episode_manager_train)
            episode_managers_test.append(episode_manager_test)
            self.episode_managers_imagination.append(episode_manager_imagination)

        def create_env(num_envs, pid):
            env_fn = partial(make_unity_gym, size=env_cfg.size)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn, pid=pid)

        if self.cfg.training_settings.should:
            train_env = create_env(train_cfg.collector_train.num_env, pid=self.cfg.collector_train.pid)
            self.train_datasets = []
            for agent_id in range(self.agent_num):
               self.train_datasets.append(EpisodesDatasetRamMonitoring(**col_cfg.train))
            self.train_collector = Collector(train_env, self.train_datasets, episode_managers_train)

        if self.cfg.evaluation_settings.should:
            test_env = create_env(train_cfg.collector_test.num_env, pid=self.cfg.collector_test.pid)
            self.test_datasets = []
            for agent_id in range(self.agent_num):
                self.test_datasets.append(EpisodesDataset(**col_cfg.test))
            self.test_collector = Collector(test_env, self.test_datasets, episode_managers_test)

        assert self.cfg.training_settings.should or self.cfg.evaluation_settings.should
        env = train_env if self.cfg.training_settings.should else test_env


        tokenizer = Tokenizer(
            vocab_size=tok_cfg.tokenizer.vocab_size, 
            embed_dim=tok_cfg.tokenizer.embed_dim,
            encoder=Encoder(EncoderDecoderConfig(**tok_cfg.tokenizer.encoder)),
            decoder=Decoder(EncoderDecoderConfig(**tok_cfg.tokenizer.decoder))
        )

        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=TransformerConfig(**worldmodel_cfg))
        actor_critic = ActorCritic(**ac_cfg, act_vocab_size=env.num_actions)

        tokenizer_state_dict = torch.load("tokenizer.pt")
        a2c_state_dict = torch.load("alg_jooyeon.pt")

        tokenizer.load_state_dict(clean_state_dict(tokenizer_state_dict, "_orig_mod.module."))
        actor_critic.load_state_dict(clean_state_dict(a2c_state_dict, "_orig_mod.module."))


        agent = Agent(tokenizer, world_model, actor_critic).to(self.device)
        agent = torch.compile(agent)

        optimizer_tokenizer = torch.optim.Adam(agent.tokenizer.parameters(), lr=train_cfg.training_settings.learning_rate)
        optimizer_world_model = configure_optimizer(agent.world_model, train_cfg.training_settings.learning_rate, train_cfg.training_settings.world_model.weight_decay)
        optimizer_actor_critic = torch.optim.Adam(agent.actor_critic.parameters(), lr=train_cfg.training_settings.learning_rate)


        for agent_id in range(self.agent_num):
            print(f"------------- Build Agent and Optimizer(id : {agent_id}) -------------")
            self.agents.append(agent)

            print(f'{sum(p.numel() for p in agent.tokenizer.parameters())} parameters in agent.tokenizer')
            print(f'{sum(p.numel() for p in agent.actor_critic.parameters())} parameters in agent.actor_critic')
            print(f'{sum(p.numel() for p in agent.world_model.parameters())} parameters in agent.world_model')


            self.optimizers_tokenizer.append(optimizer_tokenizer)
            self.optimizers_world_model.append(optimizer_world_model)
            self.optimizers_actor_critic.append(optimizer_actor_critic)
            if train_cfg.common.resume:
                self.load_checkpoint(agent_id)

    def run(self) -> None:

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            if self.cfg.training_settings.should:
                if epoch <= self.cfg.collector_train.stop_after_epochs:
                    to_log += self.train_collector.collect(self.agents, epoch, **self.cfg.collector_train.config)
                to_log += self.train_agent(epoch)

            if self.cfg.evaluation_settings.should and (epoch % self.cfg.evaluation_settings.every == 0):
                for agents_id in range(self.agent_num):
                    self.test_datasets[agents_id].clear()
                to_log += self.test_collector.collect(self.agents, epoch, **self.cfg.collector_test.config)
                to_log += self.eval_agent(epoch)

            if self.cfg.training_settings.should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})

        self.finish()
        print("end!")

    def train_agent(self, epoch: int) -> None:
        for agent in self.agents:
            agent.train()
            agent.zero_grad()
        logs = []

        for agent_id in range(self.agent_num):
            metrics_tokenizer, metrics_world_model, metrics_actor_critic = {}, {}, {}

            cfg_tokenizer = self.cfg.training_settings.tokenizer
            cfg_world_model = self.cfg.training_settings.world_model
            cfg_actor_critic = self.cfg.training_settings.actor_critic

            w = self.cfg.training_settings.sampling_weights

            if epoch > cfg_tokenizer.start_after_epochs:
                metrics_tokenizer = self.train_component(self.agents[agent_id].tokenizer, self.optimizers_tokenizer[agent_id], sequence_length=1, sample_from_start=True, sampling_weights=w, agent_id=agent_id, **cfg_tokenizer)
            self.agents[agent_id].tokenizer.eval()

            if epoch > cfg_world_model.start_after_epochs:
                metrics_world_model = self.train_component(self.agents[agent_id].world_model, self.optimizers_world_model[agent_id], sequence_length=self.cfg.common.sequence_length, sample_from_start=True, sampling_weights=w, tokenizer=self.agents[agent_id].tokenizer, agent_id=agent_id, **cfg_world_model)
            self.agents[agent_id].world_model.eval()

            if epoch > cfg_actor_critic.start_after_epochs:
                metrics_actor_critic = self.train_component(self.agents[agent_id].actor_critic, self.optimizers_actor_critic[agent_id], sequence_length=1 + self.cfg.training_settings.actor_critic.burn_in, sample_from_start=False, sampling_weights=w, tokenizer=self.agents[agent_id].tokenizer, world_model=self.agents[agent_id].world_model, agent_id=agent_id, **cfg_actor_critic)
            self.agents[agent_id].actor_critic.eval()
            logs.append({f'agent{agent_id}/epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_actor_critic})
        return logs

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int, batch_num_samples: int, grad_acc_steps: int, max_grad_norm: Optional[float], sequence_length: int, sampling_weights: Optional[Tuple[float]], sample_from_start: bool, agent_id, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        for _ in tqdm(range(steps_per_epoch), desc=f"Training {str(component)} (AgentID : {agent_id})", file=sys.stdout):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):
                batch = self.train_datasets[agent_id].sample_batch(batch_num_samples, sequence_length, sampling_weights, sample_from_start)
                batch = self._to_device(batch)

                losses = component.compute_loss(batch, **kwargs_loss) / grad_acc_steps
                loss_total_step = losses.loss_total
                loss_total_step.backward()
                loss_total_epoch += loss_total_step.item() / steps_per_epoch

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"agent{agent_id}/{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

            optimizer.step()

        metrics = {f'agent{agent_id}/{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        for agent in self.agents:
            agent.eval()

        for agent_id in range(self.agent_num):
            metrics_tokenizer, metrics_world_model = {}, {}

            cfg_tokenizer = self.cfg.evaluation_settings.tokenizer
            cfg_world_model = self.cfg.evaluation_settings.world_model
            cfg_actor_critic = self.cfg.evaluation_settings.actor_critic

            if epoch > cfg_tokenizer.start_after_epochs:
                metrics_tokenizer = self.eval_component(self.agents[agent_id].tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=1, agent_id=agent_id)

            if epoch > cfg_world_model.start_after_epochs:
                metrics_world_model = self.eval_component(self.agents[agent_id].world_model, cfg_world_model.batch_num_samples, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agents[agent_id].tokenizer, agent_id=agent_id)

            if epoch > cfg_actor_critic.start_after_epochs:
                self.inspect_imagination(epoch)

            if cfg_tokenizer.save_reconstructions:
                batch = self._to_device(self.test_datasets[agent_id].sample_batch(batch_num_samples=3, sequence_length=self.cfg.common.sequence_length))
                make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agents[agent_id].tokenizer, agent_id=agent_id)

        return [metrics_tokenizer, metrics_world_model]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, agent_id, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
        for batch in self.test_datasets[agent_id].traverse(batch_num_samples, sequence_length):
            batch = self._to_device(batch)

            losses = component.compute_loss(batch, **kwargs_loss)
            loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"agent{agent_id}/{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            pbar.update(1)

        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'agent{agent_id}/{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def inspect_imagination(self, epoch: int) -> None:
        mode_str = 'imagination'
        for agent_id in range(self.agent_num):
            batch = self.test_datasets[agent_id].sample_batch(batch_num_samples=self.episode_managers_imagination[agent_id].max_num_episodes, sequence_length=1 + self.cfg.training_settings.actor_critic.burn_in, sample_from_start=False)
            outputs = self.agents[agent_id].actor_critic.imagine(self._to_device(batch), self.agents[agent_id].tokenizer, self.agents[agent_id].world_model, horizon=self.cfg.evaluation_settings.actor_critic.horizon, show_pbar=True)

            to_log = []
            for i, (o, a, r, d) in enumerate(zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(), outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
                episode = Episode(o, a, r, d, torch.ones_like(d))
                episode_id = (epoch - 1 - self.cfg.training_settings.actor_critic.start_after_epochs) * outputs.observations.size(0) + i
                self.episode_managers_imagination[agent_id].save(episode, episode_id, epoch)

                metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
                metrics_episode['episode_num'] = episode_id
                metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.agents[agent_id].world_model.act_vocab_size)
                to_log.append({f'agent{agent_id}/{mode_str}/{k}': v for k, v in metrics_episode.items()})

        return to_log

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(epoch, self.ckpt_dir / 'epoch.pt')

        for agent_id, agent in enumerate(self.agents):
            path = self.ckpt_dir / f'agent{agent_id}'
            torch.save(agent.state_dict(), path / 'last.pt')
            if not save_agent_only:
                torch.save({
                    "optimizer_tokenizer": self.optimizers_tokenizer[agent_id].state_dict(),
                    "optimizer_world_model": self.optimizers_world_model[agent_id].state_dict(),
                    "optimizer_actor_critic": self.optimizers_actor_critic[agent_id].state_dict(),
                }, path / 'optimizer.pt')
                ckpt_dataset_dir = path / 'dataset'
                ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
                self.train_datasets[agent_id].update_disk_checkpoint(ckpt_dataset_dir)
                if self.cfg.evaluation_settings.should:
                    torch.save(self.test_datasets[agent_id].num_seen_episodes, self.ckpt_dir / f'agent{agent_id}' / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self, agent_id) -> None:
        assert self.ckpt_dir.is_dir()
        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        self.agents[agent_id].load(self.ckpt_dir / f'agent{agent_id}' / 'last.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / f'agent{agent_id}' / 'optimizer.pt', map_location=self.device)
        self.optimizers_tokenizer[agent_id].load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizers_world_model[agent_id].load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizers_actor_critic[agent_id].load_state_dict(ckpt_opt['optimizer_actor_critic'])
        self.train_datasets[agent_id].load_disk_checkpoint(self.ckpt_dir / f'agent{agent_id}' / 'dataset')
        if self.cfg.evaluation_settings.should:
            self.test_datasets[agent_id].num_seen_episodes = torch.load(self.ckpt_dir / f'agent{agent_id}' / 'num_seen_episodes_test_dataset.pt')
        print(f'Successfully loaded model, optimizer and {len(self.train_datasets[agent_id])} episodes from {self.ckpt_dir.absolute()}.(Agent Id : {agent_id})')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    def finish(self) -> None:
        wandb.finish()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()