from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gym_envs import TinyEnv
from gymnasium.envs.registration import register
import argparse
from pathlib import Path
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train_tiny_agent(args):
    # The idea is for the agent to auto tune the pid controller online.
    # This doesnt achieve any sort of stability. Maybe the pid tunings/action should be updated once and frozen for the remainder of the episode
    register(
        id="TinyEnv",
        entry_point=TinyEnv,
        max_episode_steps=999,
    )

    segments = []
    data_path = Path(args.data_path)
    if data_path.is_file():
        segments.append(str(data_path))
    elif data_path.is_dir():
        for data_file in data_path.iterdir():
            segments.append(str(data_file))
            if len(segments) >= args.num_segs:
                break 
    train_split = int(len(segments) * 0.8)

    # PPO
    vec_env = make_vec_env("TinyEnv", n_envs=args.train_n_envs, env_kwargs=dict(model_path=args.model_path, segments=segments[:train_split], debug = args.debug, on_pid=True))
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    # Separate evaluation env
    vec_env_eval = make_vec_env("TinyEnv", n_envs=args.train_n_envs, env_kwargs=dict(model_path=args.model_path, segments=segments[train_split:], debug = args.debug, on_pid=True))
    vec_env_eval = VecNormalize(vec_env_eval, norm_obs=True, norm_reward=True)

    eval_callback = EvalCallback(vec_env_eval, eval_freq=args.train_eval_freq, best_model_save_path="./logs/ppo_pid/", log_path="./logs/ppo_pid/", deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=args.train_save_freq, save_path="./logs/ppo_pid /", name_prefix="rl_model_checkpoint", save_vecnormalize=True)
   
    model = PPO(
            policy = 'MlpPolicy',
            env = vec_env,
            n_epochs = 4,
            n_steps = 1024,
            batch_size = 64,
            gamma = 0.9999,
            gae_lambda = 0.98,
            ent_coef = 0.01,
            verbose = 1,
            tensorboard_log="./logs/ppo_pid/",          
            )
    model.learn(total_timesteps=args.train_total_timesteps, callback=[eval_callback, checkpoint_callback], tb_log_name="train_run",)    
    model.save("ppo_pid_tiny")
    model = PPO.load("./logs/ppo_pid/best_model")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        for i,d in enumerate(dones):
            if d and 'costs' in info[i]:
              costs = info[i]['costs']
              print(f"\nAverage lataccel_cost: {costs['lataccel_cost']:>6.4}, average jerk_cost: {costs['jerk_cost']:>6.4}, average total_cost: {costs['total_cost']:>6.4}")
            if 'pid' in info[i]:
                tunings = info[i]['pid']
                print(f"\nPID Controller Tunings: {tunings[0]:>6.4}, {tunings[1]:>6.4}, {tunings[2]:>6.4}")      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_segs", type=int, default=100)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--train_n_envs", type=int, default=1)
    parser.add_argument("--train_total_timesteps", type=int, default=1.75e6)
    parser.add_argument("--train_save_freq", type=int, default=5e5)
    parser.add_argument("--train_eval_freq", type=int, default=5000)
    args = parser.parse_args()  
    train_tiny_agent(args)



