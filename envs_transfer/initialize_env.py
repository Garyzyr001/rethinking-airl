import gym

from .ant_env import AntEnv
from .point_maze_env import PointMazeEnv


def make_env(env_name, inference_fn=None):
    """
    Param:
        env_name: Name of the OpenAI gym environment.
    """
    if env_name == 'PointMaze-Left':
        env = PointMazeEnv(direction=0, inference_fn=inference_fn)
    elif env_name == 'PointMaze-Right':
        env = PointMazeEnv(direction=1)
    elif env_name == 'PointMaze-Double':
        env = PointMazeEnv(direction=2)
    elif env_name == 'PointMaze-Multi':
        env = PointMazeEnv(direction=3, inference_fn=inference_fn)
    elif env_name == 'Ant':
        env = AntEnv(gear=30, amputated=False)
    elif env_name == 'Ant-Disabled':
        env = AntEnv(gear=30, amputated=True, inference_fn=inference_fn)
    elif env_name == 'Ant-Lengthened':
        env = AntEnv(gear=150, big=True, inference_fn=inference_fn)
    else:
        env = gym.make(env_name)

    return env
