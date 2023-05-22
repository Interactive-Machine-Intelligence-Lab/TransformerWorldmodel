from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from envs.wrapper import ResizeObsWrapper, RewardWrapper
from mlagents_envs.environment import UnityEnvironment


def make_unity_gym(pid=0, size=64):
    
    worker_id = pid
    print('Making Workder Id :', worker_id)
    env = UnityEnvironment('envs/Red_Block/Red_Block', worker_id=worker_id)
    env = UnityToGymWrapper(env, uint8_visual=True)
    env = ResizeObsWrapper(env, (size, size))
    env = RewardWrapper(env)
    print('Created! Workder Id :', worker_id)
    return env