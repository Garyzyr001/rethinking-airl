from .ppo import PPO, PPOExpert
from .airl import AIRL
from .sac import SAC, SACExpert
from .airl_sac import AIRLSAC
from .td3 import TD3
from .airl_td3 import AIRLTD3
from .transfer_ppo import transfer_PPO
from .transfer_sac import transfer_SAC


ALGOS = {
    'ppo': PPO, 
    'airl': AIRL, 
    'sac': SAC, 
    'airl_sac': AIRLSAC,
    'td3': TD3,
    'airl_td3': AIRLTD3, 
    'transfer_ppo': transfer_PPO, 
    'transfer_sac': transfer_SAC, 
}
