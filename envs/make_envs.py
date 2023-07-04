from envs.wrapper import MultiAgentResizeObsWrapper, MultiAgentRewardWrapper
from mlagents_envs.environment import UnityEnvironment
from .multiagent_wrapper import MultiUnityWrapper
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from config import *

def make_unity_gym(pid=0, size=64):
    path = env_cfg.env_path
    worker_id = pid
    print('Making Workder Id :', worker_id)
    env = UnityEnvironment(path, worker_id=worker_id)
    #env = default_registry["VisualPushBlock"].make(worker_id=worker_id, log_folder='logs')
    env = MultiUnityWrapper(env, uint8_visual=True)
    env = MultiAgentResizeObsWrapper(env, (size, size))
    env = MultiAgentRewardWrapper(env)
    print('Created! Workder Id :', worker_id)
    return env