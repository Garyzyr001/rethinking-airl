from .ppo import PPO, PPOExpert
from .airl import AIRL
from .sac import SAC, SACExpert
from .airl_sac import AIRLSAC
from .transfer_ppo import transfer_PPO
from .transfer_sac import transfer_SAC


ALGOS = {
    'ppo': PPO, 
    'airl': AIRL, 
    'sac': SAC, 
    'airl_sac': AIRLSAC,
    'transfer_ppo': transfer_PPO, 
    'transfer_sac': transfer_SAC, 
}
