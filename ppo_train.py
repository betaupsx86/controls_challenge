from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, VectorizedActionNoise, NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from gym_envs import TinyEnv
from gymnasium.envs.registration import register
import argparse
from pathlib import Path


def train_tiny_agent(args):
    register(
        id="TinyEnv",
        entry_point=TinyEnv,
        max_episode_steps=999,
        reward_threshold=90.0,
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
    n_envs = 1
    vec_env = make_vec_env("TinyEnv", n_envs=n_envs, env_kwargs=dict(model_path=args.model_path, segments=segments[:train_split], debug = args.debug))
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    # Separate evaluation env
    vec_env_eval = make_vec_env("TinyEnv", n_envs=n_envs, env_kwargs=dict(model_path=args.model_path, segments=segments[train_split:], debug = args.debug))
    vec_env_eval = VecNormalize(vec_env_eval, norm_obs=True, norm_reward=False)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(vec_env_eval, best_model_save_path="./logs/ppo/",
                                log_path="./logs/ppo/", eval_freq=3000,
                                deterministic=True, render=False)

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
    model.learn(total_timesteps=15000, callback=eval_callback,)    
    del model
    model = PPO.load("./logs/ppo/best_model")
    model.save("ppo_tiny")

    # DDPG
    # n_envs = 1
    # vec_env = make_vec_env("TinyEnv", n_envs=n_envs, env_kwargs=dict(model_path=args.model_path, data_path=args.data_path, num_segs=args.num_segs, debug = args.debug))
    # n_actions = vec_env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # vec_action_noise = VectorizedActionNoise(action_noise, n_envs=n_envs)
    # model = DDPG(
    #         env = vec_env,
    #         policy = 'MlpPolicy',
    #         gamma = 0.98,
    #         buffer_size = 200000,
    #         learning_starts = 10000,
    #         action_noise = vec_action_noise,
    #         learning_rate = 1e-3,
    #         gradient_steps = 1,
    #         train_freq = 1,
    #         policy_kwargs = dict(net_arch=[400, 300]),
    #         verbose = 1,
    #         )
    # model.learn(total_timesteps=30000)
    # model.save("ddpg_tiny")
    # del model # remove to demonstrate saving and loading
    # model = DDPG.load("ddpg_tiny")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        if dones:
            if 'costs' in info[0]:
              costs = info[0]['costs']
              print(f"\nAverage lataccel_cost: {costs['lataccel_cost']:>6.4}, average jerk_cost: {costs['jerk_cost']:>6.4}, average total_cost: {costs['total_cost']:>6.4}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_segs", type=int, default=100)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    
    train_tiny_agent(args)

