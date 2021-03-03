import growspace
from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import gym
import numpy as np
import torch
from stable_baselines import DDPG
from stable_baselines import TRPO
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.policies import MlpPolicy

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy


def render_growspace_with_ddpg():
    seed = 123
    num_processes = 1
    gamma = 0.99
    log_dir = "."
    custom_gym = "growspace"
    recurrent_policy = False
    cuda = True
    device = torch.device("cuda:0" if cuda else "cpu")

    envs = make_vec_envs("GrowSpaceEnv-Continuous-v0", seed, num_processes, gamma, log_dir, device, False, custom_gym)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space, base_kwargs={'recurrent': recurrent_policy})
    actor_critic.to(device)

    eval_recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    obs = envs.reset()
    while True:
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True
            )
        obs, rewards, dones, info = envs.step(action)
        envs.render()


def render_growspace_with_ppo():
    seed = 123
    num_processes = 1
    gamma = 0.99
    log_dir = "."
    custom_gym = "growspace"
    recurrent_policy = False
    cuda = True
    device = torch.device("cuda:0" if cuda else "cpu")

    envs = make_vec_envs("GrowSpaceEnv-Continuous-v0", seed, num_processes, gamma, log_dir, device, False, custom_gym)

    # actor_critic = Policy(envs.observation_space.shape, envs.action_space, base_kwargs={'recurrent': recurrent_policy})
    actor_critic.to(device)

    eval_recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    obs = envs.reset()
    while True:
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True
            )
        obs, rewards, dones, info = envs.step(action)
        envs.render()


def render_growspace_with_trpo():
    env = gym.make('GrowSpaceEnv-Control-v0')
    model = TRPO(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=2500)
    # model.save("trpo_cartpole")
    #
    # del model  # remove to demonstrate saving and loading

    model = TRPO.load("trpo_cartpole")

    obs = env.reset()
    for t in range(150):
        print(t)
        # while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        # if dones:
        #     env.reset()
        env.render()


def render_to_gif():
    def save_frames_as_gif(frames, path='./', filename='growspace_with_trpo.gif'):
        # Mess with this to change frame size
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        anim.save(path + filename, writer='imagemagick', fps=60)

    env = gym.make('GrowSpaceEnv-Control-v0')
    model = TRPO(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=2500)
    # model.save("trpo_cartpole")

    # del model  # remove to demonstrate saving and loading

    model = TRPO.load("trpo_cartpole")

    frames = []
    obs = env.reset()
    for _ in range(150):
        # while True:
        frames.append(env.render(mode="rgb_array"))

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # if done:
        #     break
        # env.render()

    env.close()
    save_frames_as_gif(frames)


if __name__ == '__main__':
    # render_growspace_with_ppo()
    # render_growspace_with_trpo()
    # render_to_gif()
    render_growspace_with_ddpg()