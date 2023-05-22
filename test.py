from envs import make_unity_gym
from pyvirtualdisplay.display import Display

with Display() as disp:
  env = make_unity_gym()
  env.close()