from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, VectorizedActionNoise, NormalActionNoise
from gym_envs import TinyEnv
from gymnasium.envs.registration import register
import argparse


def train_tiny_agent(args):
    register(
        id="TinyEnv",
        entry_point=TinyEnv,
        max_episode_steps=999,
        reward_threshold=90.0,
    )
    
    # The idea is for the agent to learn the pid tuning online.
    # This is absolutely unstable. Maybe freeze pid tuning for several episodes?
    n_envs = 1
    vec_env = make_vec_env("TinyEnv", n_envs=n_envs, env_kwargs=dict(on_pid=True, model_path=args.model_path, data_path=args.data_path, num_segs=args.num_segs, debug = args.debug))
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    model = PPO(
            policy = 'MlpPolicy',
            env = vec_env,
            learning_rate = 7.77e-05,
            n_steps = 8,
            batch_size = 256,
            n_epochs = 10,
            gamma = 0.9999,
            gae_lambda = 0.9,
            clip_range = 0.1,
            ent_coef = 0.00429,
            vf_coef = 0.19,
            max_grad_norm = 5,
            use_sde = True,
            policy_kwargs = dict(log_std_init=-3.29, ortho_init=False),
            verbose = 1,
            )
    model.learn(total_timesteps=30000)
    model.save("ppo_pid_tiny")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        if dones:
            if 'costs' in info[0]:
              costs = info[0]['costs']
              print(f"\nAverage lataccel_cost: {costs['lataccel_cost']:>6.4}, average jerk_cost: {costs['jerk_cost']:>6.4}, average total_cost: {costs['total_cost']:>6.4}")
        else:
            if 'pid' in info[0]:
                tunings = info[0]['pid']
                print(f"\nPID Controller Tunings: {tunings[0]:>6.4}, {tunings[1]:>6.4}, {tunings[2]:>6.4}")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_segs", type=int, default=100)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    
    train_tiny_agent(args)

