from mlagents_envs.registry import default_registry
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from envs.wrapper import ResizeObsWrapper

def make_unity_gym(pid=0, size=64):
    env_name = "VisualPushBlock"
    worker_id = pid
    print('Making Workder Id :', worker_id)
    env = default_registry[env_name].make(worker_id=worker_id)
    env = UnityToGymWrapper(env, uint8_visual=True)
    env = ResizeObsWrapper(env, (size, size))
    print('Created! Workder Id :', worker_id)
    return env
