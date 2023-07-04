from functools import partial 
from pathlib import Path
import torch

from agent import Agent
from envs import SingleProcessEnv, WorldModelEnv
from game import AgentEnv, EpisodeReplayEnv, Game, PlayerEnv
from models.actor_critic import ActorCritic
from models.world_model import WorldModel, TransformerConfig
from models.tokenizer import Tokenizer, Encoder, Decoder, EncoderDecoderConfig

from envs import make_unity_gym

from config import *

def main():
    device = torch.device(replay_cfg.replay_device)
    mode_dict = {
        1 : "episode_replay",
        2 : 'agent_in_env',
        3 : 'agent_in_world_model',
        4 : 'play_in_world_model',
        5 : 'play_in_env',
    }
    mode = mode_dict[replay_cfg.mode]
    agent_id = replay_cfg.agent_id

    env_fn = partial(make_unity_gym, size=64)
    test_env = SingleProcessEnv(env_fn, [10])

    if mode.startswith('agent_in_'):
        h, w, _ = test_env.env.unwrapped.observation_space[0].shape
    else:
        h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]
    
    if mode == 'episode_replay':
        env = EpisodeReplayEnv(episode_dir=Path(replay_cfg.replay_path))
        keymap = 'episode_replay'

    else:
        agents = []
        # for id in range(env_cfg.agent_num):
        #     tokenizer = Tokenizer(
        #         vocab_size=tok_cfg.tokenizer.vocab_size, 
        #         embed_dim=tok_cfg.tokenizer.embed_dim,
        #         encoder=Encoder(EncoderDecoderConfig(**tok_cfg.tokenizer.encoder)),
        #         decoder=Decoder(EncoderDecoderConfig(**tok_cfg.tokenizer.decoder))
        #     )        
        #     world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions, config=TransformerConfig(**worldmodel_cfg))
        #     actor_critic = ActorCritic(**ac_cfg.actor_critic, act_vocab_size=test_env.num_actions)
        #     agent = Agent(tokenizer, world_model, actor_critic).to(device)
        #     agent.load(Path(replay_cfg.agent_checkpoint_path) / f'agent{id}' / 'last.pt', device)

        #     agents.append(agent)

        if mode == 'play_in_world_model':
            env = WorldModelEnv(agent_id=agent_id, tokenizer=agents[agent_id].tokenizer, world_model=agents[agent_id].world_model, device=device, env=env_fn())
            keymap = 'default'
        
        elif mode == 'agent_in_env':
            env = AgentEnv(agent, test_env, do_reconstruction=replay_cfg.do_reconstruction)
            keymap = 'empty'
            if replay_cfg.do_reconstruction:
                size[1] *= 3

        elif mode == 'agent_in_world_model':
            wm_env = WorldModelEnv(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device, env=env_fn(), agent_id=agent_id)
            env = AgentEnv(agent, wm_env, do_reconstruction=False)
            keymap = 'empty'

        elif mode == 'play_in_env':
            env = PlayerEnv(env=test_env, agent_id=replay_cfg.agent_id, agent_num=env_cfg.agent_num)
            keymap = 'default'

    game = Game(env, keymap_name=keymap, size=size, fps=replay_cfg.fps, verbose=bool(replay_cfg.header), record_mode=bool(replay_cfg.save_mode))
    game.run()


if __name__ == "__main__":
    main()
