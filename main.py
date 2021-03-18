import logging

try:
    from comet_ml import Experiment

    comet_loaded = True
except ImportError:
    comet_loaded = False
import getpass
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.backends.cudnn
from comet_ml import Experiment
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2 ** 84)
import cv2
from comet_ml import Experiment


def main():
    args = get_args()

    wandb.init(project='ppo', entity='growspace')
    # wandb.config()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(
        args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False, args.custom_gym
    )

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy}
    )
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm
        )
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            optimizer="adam"
        )
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = []
    episode_length = []
    episode_branches = []
    episode_branch1 = []
    episode_branch2 = []
    episode_light_width = []
    episode_light_move = []
    episode_success = []
    episode_light_position = []
    episode_beam_width = []

    episode_success_rate = deque(maxlen=100)
    episode_total = 0

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    x = 0
    step_logger_counter = 0

    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        # new_branches = []
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                # wandb.log({"Actions": wandb.Histogram(action.detach().cpu())}, step=step_logger_counter)
                print(f"action: {action.detach().cpu()}")
                step_logger_counter += 1

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            # episode_light_position.append(action[:, 0])
            # episode_beam_width.append(action[:, 1])

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_length.append(info['episode']['l'])

                    # if j % args.log_interval == 0 and len(episode_rewards) > 1:
                    #     wandb.log({"Episode Reward": info['episode']['r'].item()}, step=step_logger_counter)
                    #     step_logger_counter += 1

                if 'new_branches' in info.keys():
                    episode_branches.append(info['new_branches'])

                if 'new_b1' in info.keys():
                    episode_branch1.append(info['new_b1'])

                if 'new_b2' in info.keys():
                    episode_branch2.append(info['new_b2'])

                if 'light_width' in info.keys():
                    episode_light_width.append(info['light_width'])

                if 'light_move' in info.keys():
                    episode_light_move.append(info['light_move'])
                    # print("what is new branches", new_branches)

                if 'success' in info.keys():
                    episode_success.append(info['success'])

                if j == x:
                    if 'img' in info.keys():
                        img = info['img']
                        path = './hittiyas/growspaceenv_braselines/scripts/imgs/'
                        cv2.imwrite(os.path.join(path, 'step' + str(step) + '.png'), img)
                    x += 1000

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        # print("before")
        # episode_branches.append(np.asarray([[np.mean(new_branches)]]))
        # print("after")
        # print(episode_branches)
        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError as error:
                logging.warning(f"exception: {error}")

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            wandb.log({"Reward Min": np.min(episode_rewards)}, step=total_num_steps)
            wandb.log({"Reward Mean": np.mean(episode_rewards)}, step=total_num_steps)
            wandb.log({"EPISODE REWARD": episode_rewards}, step=total_num_steps)
            wandb.log({"Reward Max": np.max(episode_rewards)}, step=total_num_steps)
            wandb.log({"Number of Mean New Branches": np.mean(episode_branches)}, step=total_num_steps)
            wandb.log({"Number of Max New Branches": np.max(episode_branches)}, step=total_num_steps)
            wandb.log({"Number of Min New Branches": np.min(episode_branches)}, step=total_num_steps)
            wandb.log({"Number of Mean New Branches of Plant 1": np.mean(episode_branch1)}, step=total_num_steps)
            wandb.log({"Number of Mean New Branches of Plant 2": np.mean(episode_branch2)}, step=total_num_steps)
            wandb.log({"Number of Total Displacement of Light": np.sum(episode_light_move)}, step=total_num_steps)
            wandb.log({"Mean Light Displacement": np.mean(episode_light_move)}, step=total_num_steps)
            wandb.log({"Mean Light Width": np.mean(episode_light_width)}, step=total_num_steps)
            wandb.log({"Number of Steps in Episode with Tree is as close as possible": np.sum(episode_success)},
                      step=total_num_steps)
            wandb.log({"Number of Total New Branches": np.sum(episode_branches)}, step=total_num_steps)
            wandb.log({"Episode Length Mean ": np.mean(episode_length)}, step=total_num_steps)
            wandb.log({"Episode Length Min": np.min(episode_length)}, step=total_num_steps)
            wandb.log({"Episode Length Max": np.max(episode_length)}, step=total_num_steps)
            wandb.log({"Entropy": dist_entropy}, step=total_num_steps)
            wandb.log({"Displacement of Light Position": wandb.Histogram(episode_light_position)}, step=total_num_steps)
            # experiment.log_histogram_3d(name="Displacement Beam Width", values=episode_beam_width,
            #                             step=total_num_steps)
            #
            # experiment.log_histogram_3d(name="Displacement of Light Move", values=episode_light_move,
            #                             step=total_num_steps)
            # experiment.log_histogram_3d(name="Displacement Light Width", values=episode_light_width,
            #                             step=total_num_steps)

            print(f"Updates: {j}, timesteps {total_num_steps}, FPS: {int(total_num_steps / (end - start))}, "
                  f"Last reward: {len(episode_rewards)}, \n"
                  f"training episodes: mean/median reward {np.mean(episode_rewards)}/{np.median(episode_rewards)}, \n"
                  f"min/max reward {np.min(episode_rewards)}/{np.max(episode_rewards)}, \n"
                  f"dist_entropy: {dist_entropy}, value_loss: {value_loss}, action_loss: {action_loss}")

            # gif, gif_filepath = create_render_for_comet(args, actor_critic)
            # experiment.log_asset(gif_filepath)

            episode_rewards.clear()
            episode_length.clear()
            episode_branches.clear()
            episode_branch1.clear()
            episode_branch2.clear()
            episode_light_width.clear()
            episode_light_move.clear()
            episode_success.clear()
            episode_light_position.clear()
            episode_light_width.clear()

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
