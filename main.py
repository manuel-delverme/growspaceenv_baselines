try:
    from comet_ml import Experiment
    comet_loaded = True
except ImportError:
    comet_loaded = False
import os
import time
from collections import deque

import numpy as np
import torch

import growspace
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import getpass

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2 ** 84)
import cv2
from comet_ml import Experiment

def main():
    args = get_args()

    if comet_loaded:
        experiment = Experiment(
            api_key="WRmA8ms9A78K85fLxcv8Nsld9",
            project_name="growspace2021",
            workspace="yasmeenvh")
        experiment.add_tag(getpass.getuser())
        experiment.set_name(args.comet)
        for key, value in vars(args).items():
            experiment.log_parameter(key, value)
    else:
        experiment = None
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

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, args.custom_gym)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
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
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_length = deque(maxlen=10)
    episode_branches = deque(maxlen=10)
    episode_branch1 = deque(maxlen=10)
    episode_branch2 = deque(maxlen=10)
    episode_light_width = deque(maxlen=10)
    episode_light_move = deque(maxlen=10)
    episode_success = deque(maxlen=10)
    # episode_light_move = deque(maxlen=10)
    # new_branches = []
    episode_success_rate = deque(maxlen=100)
    episode_total = 0

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    # print("what are the num_updates",num_updates)
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
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            if experiment is not None:
                experiment.log_metric("Episode Reward During Training", reward.item(), step=step_logger_counter)
                step_logger_counter += 1

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_length.append(info['episode']['l'])

                if 'new_branches' in info.keys():
                    episode_branches.append(info['new_branches'])
                    # print("what is in new branches", info['new_branches'])
                    # print("type of data:", type(info['new_branches']))

                    # print("what is new branches", new_branches)
                if 'new_b1' in info.keys():
                    episode_branch1.append(info['new_b1'])

                if 'new_b2' in info.keys():
                    episode_branch2.append(info['new_b2'])

                if 'light_width' in info.keys():
                    episode_light_width.append(info['light_width'])
                    # print("what is new branches", new_branches)

                if 'light_move' in info.keys():
                    episode_light_move.append(info['light_move'])
                    # print("what is new branches", new_branches)

                if 'success' in info.keys():
                    episode_success.append(info['success'])
                    # print("what is new branches", new_branches)

                if j == x:
                    if 'img' in info.keys():
                        img = info['img']
                        path = './hittiyas/growspaceenv_braselines/scripts/imgs/'
                        cv2.imwrite(os.path.join(path, 'step' + str(step) + '.png'), img)
                    x += 1000

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        #print("before")
        #episode_branches.append(np.asarray([[np.mean(new_branches)]]))
        #print("after")
        #print(episode_branches)

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()


        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            if experiment is not None:
                experiment.log_metric("Reward Mean", np.mean(episode_rewards), step=total_num_steps)
                experiment.log_metric("Reward Min", np.min(episode_rewards), step=total_num_steps)
                experiment.log_metric("Reward Max", np.max(episode_rewards), step=total_num_steps)
                experiment.log_metric("Number of Mean New Branches", np.mean(episode_branches), step=total_num_steps)
                experiment.log_metric("Number of Total New Branches", np.sum(episode_branches), step=total_num_steps)
                experiment.log_metric("Number of Min New Branches", np.min(episode_branches), step=total_num_steps)
                experiment.log_metric("Number of Max New Branches", np.max(episode_branches), step=total_num_steps)

                experiment.log_metric("Number of Mean New Branches of Plant 1", np.mean(episode_branch1), step=total_num_steps)
                experiment.log_metric("Number of Mean New Branches of Plant 2", np.mean(episode_branch2), step=total_num_steps)

                experiment.log_metric("Number of Total Displacement of Light", np.sum(episode_light_move), step=total_num_steps)
                experiment.log_metric("Mean Displacement of Light", np.mean(episode_light_move), step=total_num_steps)
                experiment.log_metric("Mean Light Width", np.mean(episode_light_width), step=total_num_steps)
                experiment.log_metric("Number of Steps in Episode with Tree is as close as possible", np.sum(episode_success), step=total_num_steps)
                experiment.log_metric("Episode Length Mean ", np.mean(episode_length), step=total_num_steps);
                experiment.log_metric("Episode Length Min",
                                      np.min(episode_length), step=total_num_steps)
                experiment.log_metric("Episode Length Max", np.max(episode_length), step=total_num_steps)

            # print("Number of mean branches", np.mean(episode_branches))

        if (args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
