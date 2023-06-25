
from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np
from gym import error, spaces
from mlagents_envs import logging_util
from mlagents_envs.base_env import BaseEnv, DecisionSteps, TerminalSteps, ActionTuple

class MultiUnityWrapperException(error.Error):
    """
    Any error related to the multiagent wrapper of ml-agents.
    """


logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)

GymStepResult = Tuple[np.ndarray, float, bool, Dict]
MultiStepResult = Tuple[Dict[str, np.ndarray],
                        Dict[str, float], Dict[str, bool], Dict]


class MultiUnityWrapper(gym.Env):
    """
    Provides wrapper for Unity Learning Environments, supporting multiagents.
    """
    # Implemented class. Implements: rllib.MultiEnv.
    # (not done because rllib cannot be installed on windows for now)

    def __init__(
        self,
        unity_env: BaseEnv,
        uint8_visual: bool = False,
        allow_multiple_obs: bool = False,
    ):
        """
        Environment initialization
        :param unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
        :param use_visual: Whether to use visual observation or vector observation.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param allow_multiple_obs: If True, return a list of np.ndarrays as observations with the first elements
            containing the visual observations and the last element containing the array of vector observations.
            If False, returns a single np.ndarray containing either only a single visual observation or the array of
            vector observations.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.visual_obs = None

        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: DecisionSteps = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs

        self.behaviour_names = [
            name for name in self._env.behavior_specs.keys()]

        # Check for number of agents in scene.
        self._n_agents = 0
        self._env.reset()
        self._agent_id_to_behaviour_name = {}
        self._agents_dict = {}
        for name in self.behaviour_names:
            decision_steps, _ = self._env.get_steps(name)
            self._agents_dict[name] = []
            for agent_id in decision_steps.agent_id:
                self._agent_id_to_behaviour_name[agent_id] = name
                self._agents_dict[name].append(agent_id)
                self._n_agents += 1
            self._previous_decision_step = decision_steps

        if self._get_n_vis_obs() == 0 and self._get_vec_obs_size() == 0:
            raise MultiUnityWrapperException(
                "There are no observations provided by the environment."
            )

        if not all(self._get_n_vis_obs().values()) >= 1 and uint8_visual:
            logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual

        if all(self._get_n_vis_obs().values()) + all(self._get_vec_obs_size().values()) >= 2 and not self._allow_multiple_obs:
            logger.warning(
                "The environment contains multiple observations. "
                "You must define allow_multiple_obs=True to receive them all. "
                "Otherwise, only the first visual observation (or vector observation if"
                "there are no visual observations) will be provided in the observation."
            )

        # Set observation and action spaces
        self._action_space = {}
        self._observation_space = {}
        vec_obs_size_dict = self._get_vec_obs_size()
        shape_dict = self._get_vis_obs_shape()
        for behaviour_name, group_spec in self._env.behavior_specs.items():
            # Set observations space
            if group_spec.action_spec.is_discrete():
                branches = group_spec.action_spec.discrete_branches
                if group_spec.action_spec.discrete_size == 1:
                    action_space = spaces.Discrete(branches[0])
                else:
                    raise MultiUnityWrapperException("group_spec.action_spec.discrete_size is not 1")

            else:
                raise MultiUnityWrapperException("Action space is continuous.")

            # Set observations space
            list_spaces: List[gym.Space] = []
            shapes = shape_dict[behaviour_name]
            for shape in shapes:
                if uint8_visual:
                    list_spaces.append(spaces.Box(
                        0, 255, dtype=np.uint8, shape=shape))
                else:
                    list_spaces.append(spaces.Box(
                        0, 1, dtype=np.float32, shape=shape))
            if vec_obs_size_dict[behaviour_name] > 0:
                # vector observation is last
                high = np.array([np.inf] * vec_obs_size_dict[behaviour_name])
                list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
            if self._allow_multiple_obs:
                observation_space = spaces.Tuple(list_spaces)
            else:
                observation_space = list_spaces[0]  # only return the first one

            # Assign spaces to agents
            for agent_id in self._agents_dict[behaviour_name]:
                self._observation_space[agent_id] = observation_space
                self._action_space[agent_id] = action_space

    def reset(self) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        """
        Resets the state of the environment and returns an initial observation.
        Returns: observation (object/list): the initial observation of the
        space.
        """
        self._env.reset()
        decision_steps_dict = {behaviour_name: self._env.get_steps(
            behaviour_name)[0] for behaviour_name in self.behaviour_names}
        n_agents = sum([len(decision_step)
                        for decision_step in decision_steps_dict.values()])
        self._check_agents(n_agents)
        self.game_over = False

        res: GymStepResult = self._single_step(decision_steps_dict)
        # Returns only observation
        return res[0]

    def step(self, action_dict: Dict) -> MultiStepResult:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action_dict (dict): dict of actions provided by all agents
        Returns:
            observation (dict): agents' observations of the current environment
            reward (dict) : amount of rewards returned after previous action
            done (dict): whether the episode has ended for each agent.
            info (dict): contains auxiliary diagnostic information.
        """

        if self.game_over:
            raise MultiUnityWrapperException(
                "You are calling 'step()' even though this environment has already "
                "returned done = True. You must always call 'reset()' once you "
                "receive 'done = True'."
            )

        for agent_id, action in action_dict.items():
            behaviour_name = self._agent_id_to_behaviour_name[agent_id]
            action = np.array(action).reshape((1, 1))

            action_tuple = ActionTuple()
            action_tuple.add_discrete(action)
            self._env.set_action_for_agent(behaviour_name, agent_id, action_tuple)

        self._env.step()
        decision_steps_dict, terminal_steps_dict = {}, {}

        for behaviour_name in self.behaviour_names:
            decision_step, terminal_step = self._env.get_steps(behaviour_name)
            decision_steps_dict[behaviour_name] = decision_step
            terminal_steps_dict[behaviour_name] = terminal_step

        self._check_agents(
            max(len(decision_steps_dict), len(terminal_steps_dict)))
        
        terminal_obs_dict, terminal_reward_dict, terminal_done_dict, terminal_info = {}, {}, {}, {}
        decision_obs_dict, decision_reward_dict, decision_done_dict, decision_info = {}, {}, {}, {}
        if len(terminal_step) != 0:
            # At least one agent is done
            self.game_over = True
            terminal_obs_dict, terminal_reward_dict, terminal_done_dict, terminal_info = self._single_step(terminal_steps_dict)
        else:
            decision_obs_dict, decision_reward_dict, decision_done_dict, decision_info = self._single_step(decision_steps_dict)
            terminal_reward_dict, terminal_done_dict, terminal_info = {}, {}, {}

        # Create MultiStepResult dicts
        # Episode is done: no terminal_obs
        if len(terminal_step) != 0:
            obs_dict = terminal_obs_dict
        else:
            obs_dict = decision_obs_dict
        reward_dict = {**decision_reward_dict, **terminal_reward_dict}
        done_dict = {**decision_done_dict, **terminal_done_dict}
        info_dict = {"decision_step": decision_info,
                     "terminal_step": terminal_info}

        # Game is over when all agents are done
        done_dict["__all__"] = self.game_over = (all(done_dict.values()) and len(
            done_dict.values()) == self._n_agents)
        return (obs_dict, reward_dict, done_dict, info_dict)

    def _single_step(self, info_dict: Dict[str, Tuple[DecisionSteps, TerminalSteps]]) -> GymStepResult:
        obs_dict, reward_dict, done_dict = {}, {}, {}
        vec_obs_size = self._get_vec_obs_size()
        n_vis_obs = self._get_n_vis_obs()
        

        for behaviour_name, info in info_dict.items():
            default_observation = None
            if self._allow_multiple_obs:
                visual_obs = self._get_vis_obs_list(info)
                visual_obs_list = []
                for obs in visual_obs:
                    visual_obs_list.append(self._preprocess_single(obs[0]))
                default_observation = visual_obs_list
                if vec_obs_size[behaviour_name] >= 1:
                    default_observation.append(
                        self._get_vector_obs(info))
            else:
                if n_vis_obs[behaviour_name] >= 1:
                    visual_obs = self._get_vis_obs_list(info)
                    for agent_id in info.agent_id:
                        default_observation = self._preprocess_single(visual_obs[0][agent_id])
                        obs_dict[agent_id] = default_observation
                else:
                    obs_dict.update(self._get_vector_obs(info))

            if n_vis_obs[behaviour_name] >= 1:
                visual_obs = self._get_vis_obs_list(info)
                self.visual_obs = self._preprocess_single(visual_obs[0])

            done = isinstance(info, TerminalSteps)
            for agent_id in info.agent_id:
                # Add reward and done
                agent_index = info.agent_id_to_index[agent_id]
                reward_dict[agent_id] = info.reward[agent_index]
                done_dict[agent_id] = done
        return (obs_dict, reward_dict, done_dict, info)

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        else:
            return single_visual_obs

    def _get_n_vis_obs(self) -> Dict:
        n_vis_obs_dict = {}
        for behaviour_name, group_spec in self._env.behavior_specs.items():
            result = 0
            for shape in group_spec.observation_specs:
                if len(shape.shape) == 3:
                    result += 1
            n_vis_obs_dict[behaviour_name] = result
        return n_vis_obs_dict

    def _get_vis_obs_shape(self) -> Dict[str, List[Tuple]]:
        vis_obs_shape_dict = {}
        for behaviour_name, group_spec in self._env.behavior_specs.items():
            result: List[Tuple] = []
            for shape in group_spec.observation_specs:
                if len(shape.shape) == 3:
                    result.append(shape.shape)
            vis_obs_shape_dict[behaviour_name] = result

        return vis_obs_shape_dict

    def _get_vis_obs_list(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> Dict[str, np.ndarray]:
        vector_obs_dict = {}
        for agents_obs in step_result.obs:
            if len(agents_obs.shape) == 2:
                for agent_id, obs in zip(step_result.agent_id, agents_obs):
                    vector_obs_dict[agent_id] = obs
        return vector_obs_dict

    def _get_vec_obs_size(self) -> Dict:
        vec_obs_size_dict = {}
        for behaviour_name, group_spec in self._env.behavior_specs.items():
            result = 0
            for shape in group_spec.observation_specs:
                if len(shape.shape) == 1:
                    result += shape.shape[0]
            vec_obs_size_dict[behaviour_name] = result
        return vec_obs_size_dict

    def render(self, mode="rgb_array"):
        return self.visual_obs

    def close(self) -> None:
        """
        Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        """
        Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        logger.warning("Could not seed environment")
        return

    # This method is a staticmethod in UnityToGym but here we need the number of agents (self._n_agents) in the env!
    def _check_agents(self, n_agents: int) -> None:
        if n_agents > self._n_agents:
            raise MultiUnityWrapperException(
                f"There can only be {self._n_agents} Agents in the environment but {n_agents} were detected."
            )

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        """
        Range in which rewards stand

        Returns:
            Tuple[float, float]: (-inf, inf)
        """
        return -float("inf"), float("inf")

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        'List of action space corresponding to each agent'
        return self._action_space

    @property
    def observation_space(self):
        'List of observation space corresponding to each agent'
        return self._observation_space

    # Does not exist anymore in UnityToGym (one agent) but it makes sense here.
    @property
    def number_agents(self):
        'Number of agents in the env'
        return self._n_agents
