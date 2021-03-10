import os

import matplotlib.pyplot as plt
import torch
from matplotlib import animation

from a2c_ppo_acktr.envs import make_vec_envs


def create_render_for_comet(args, actor_critic, num_processes=1):
    device = torch.device("cuda:0" if args.cuda else "cpu")

    env_gym = make_vec_envs(
        "GrowSpaceEnv-Continuous-v0",
        args.seed,
        num_processes,
        args.gamma,
        args.log_dir,
        device,
        False,
        args.custom_gym
    )

    frames = []
    obs = env_gym.reset()
    eval_recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size,
                                               device=device)
    eval_masks = torch.zeros(args.num_processes, 1, device=device)
    for _ in range(150):
        frames.append(env_gym.render(mode="rgb_array"))

        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True
            )
        obs, rewards, dones, info = env_gym.step(action)
        if dones:
            break
        # env_gym.render()

    env_gym.close()

    gif, gif_filepath = save_frames_as_gif(frames)

    return gif, gif_filepath


def save_frames_as_gif(frames, path="./", filename: str = "please_name_this_gif.gif"):
    def animate(i):
        patch.set_data(frames[i][:, :, ::-1])

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    return anim, path + filename


def make_movie_list():
    png_dir = './hittiyas/growspaceenv_braselines/scripts/imgs/'

    step = 0
    movie_files = []
    print(os.listdir(png_dir))
    for i in range(0, len(os.listdir(png_dir))):
        if os.listdir(png_dir)[i].startswith("step"):
            movie_files.append(os.listdir(png_dir)[i])

    return movie_files


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]
