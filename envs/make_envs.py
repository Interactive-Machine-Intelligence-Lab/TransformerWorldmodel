from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from envs.wrapper import ResizeObsWrapper
from mlagents_envs.environment import UnityEnvironment

def make_unity_gym(pid=0, size=64):
    worker_id = pid
    print('Making Workder Id :', worker_id)
    env = UnityEnvironment('envs/Red_Block/Red_Block', worker_id=worker_id)
    env = UnityToGymWrapper(env, uint8_visual=True)
    env = ResizeObsWrapper(env, (size, size))
    print('Created! Workder Id :', worker_id)
    return env



# from mlagents_envs.registry import default_registry
# from mlagents_envs.environment import UnityEnvironment
# from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
# from envs.wrapper import ResizeObsWrapper
# import time

# def make_unity_gym(pid=0, size=64):
#     env_name = "VisualPushBlock"
#     if pid < 10:
#       worker_id = pid + 5000
#     else:
#       worker_id = pid
#     print('Making Workder Id :', worker_id)
    
    
#     #with Display() as disp:
#     env = default_registry[env_name].make(worker_id=worker_id)
#     # env = UnityEnvironment('envs/Red_Block/Red_Block.x86_64', worker_id=worker_id, log_folder='/home/jooyeon/jeonminguk/TransformerWorldmodel/envs/logs', timeout_wait=100)
#     env = UnityToGymWrapper(env, uint8_visual=True)
#     env = ResizeObsWrapper(env, (size, size))
#     print('Created! Workder Id :', worker_id)
#     return env
