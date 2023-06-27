from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from envs.wrapper import ResizeObsWrapper, RewardWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.registry import default_registry


def make_unity_gym(pid=0, size=64):
    env_name = "VisualPushBlock"

    worker_id = pid
    print('Making Workder Id :', worker_id)
    env = UnityEnvironment('envs/pushblock/GreenBlock.x86_64', worker_id=worker_id)
    #env = default_registry[env_name].make(worker_id=worker_id, log_folder='logs')
    
    
    print('wrapping')
    env = UnityToGymWrapper(env, uint8_visual=True)
    env = ResizeObsWrapper(env, (size, size))
    env = RewardWrapper(env)
    print('Created! Workder Id :', worker_id)
    return env

if __name__ == "__main__":
  print("making envs")
  env = make_unity_gym(5500)
  print("done")
  env.close()
  print("close")
