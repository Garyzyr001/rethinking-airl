## Setup
You can install Python libraries using `pip install -r requirements.txt`. Note that you need a MuJoCo license. Please follow the instructions in [mujoco-py] (https://github.com/openai/mujoco-py) for help.

## Example
### Train expert
You can train experts using soft actor-critic (SAC) [[1,2]](#references).  

```bash
python train_expert.py --cuda --env_id PointMaze-Right --num_steps 1000000 --seed 0
```

Its seed is named "seed0-20230805-1354" for instance. 

### Collect demonstrations
You need to collect demonstrations using the trained expert's weight. Note that `--std` specifies the standard deviation of the Gaussian noise added to the action, and `--p_rand` specifies the probability the expert acts randomly. We set `std` to 0.01 not to collect too similar trajectories.

```bash
python collect_demo.py \
    --cuda --env_id PointMaze-Right \
    --weight logs/PointMaze-Right/sac/seed0-20230805-1354/model/step1000000/actor.pth \
    --buffer_size 1000000 --std 0.01 --p_rand 0.0 --seed 0
```

### Train AIRL
Experts' demonstrations are provided in `buffers/`. You can train SAC-AIRL in the source environment using the demonstrations above. For example, 

```bash
python train_airl.py \
    --cuda --env_id PointMaze-Right \
    --buffer buffers/PointMaze-Right/size1000000_std0.01_prand0.0.pth \
    --num_steps 1500000 --eval_interval 5000 --rollout_length 64 --seed 0 \
    --algo 'airl_sac' --epoch_disc 1 --epoch_policy 32 --batch_size 64 --cuda_id 0
```

Its seed is named "seed0-20231226-1522" for instance. 

### Train Reward Transfer Imitation Learning
You can re-optimize the policy in the new environment via the learned reward in the source environment. For example, 

```bash
python train_transfer_imitation.py \
    --cuda --env_id PointMaze-Left \
    --num_steps 1000000 --seed 0 \
    --algo 'transfer_sac' --airl_env_id PointMaze-Right \
    --load_seed 0 --load_time 20231226-1522 --cuda_id 0
```


## References
[[1]](https://arxiv.org/abs/1801.01290) Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).

[[2]](https://arxiv.org/abs/1812.05905) Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).


