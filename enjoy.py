import argparse
import os

# workaround to unpickle olf model files
import sys
import time
import imageio
import numpy as np
import torch
from datetime import datetime
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

sys.path.append("a2c_ppo_acktr")

parser = argparse.ArgumentParser(description="RL")
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=10, help="log interval, one log per n updates (default: 10)")
parser.add_argument(
    "--env-name", default="GrowSpaceEnv-ControlEasy-v0", help="environment to train on (default: PongNoFrameskip-v4)"
)
parser.add_argument("--custom-gym", default="growspace", help="The gym to load from")
parser.add_argument(
    "--model",
    default="/home/y/Documents/growspaceenv_baselines/scripts/GrowSpaceEnv-Control-v0.pt",
    help="directory to save agent logs (default: ./trained_models/ppo/)",
)
parser.add_argument("--non-det", action="store_true", default=False, help="whether to use a non-deterministic policy")
parser.add_argument(
    "--gif", type=int, default=0, help="If you set this value to a positive int, it will render that many images to gif"
)
# parser.add_argument("--non-det", action="store_true", default=False, help="whether to use a non-deterministic policy")
parser.add_argument("--frame-stacc", type=int, default=4, help="how many past frames are kept in memory?")
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(
    args.env_name, args.seed + 1000, 1, None, None, device="cpu", custom_gym=args.custom_gym, allow_early_resets=False
)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = torch.load(args.model, map_location="cpu")

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)
obs = env.reset()

if render_func is not None:
    mode = "human"
    if args.gif > 0:
        mode = "rgb_array"
    render_func(mode)

if args.env_name.find("Bullet") > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if p.getBodyInfo(i)[0].decode() == "torso":
            torsoId = i

frame_counter = 0
images = []
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det
        )

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    masks.fill_(0.0 if done else 1.0)

    print(f"{frame_counter}, action: {action}, reward: {reward}, done: {done}")

    frame_counter += 1

    if args.gif <= 0:
        # not writing GIF, only showing to user
        if render_func is not None:
            render_func("human")
        time.sleep(0.5)
    else:
        images.append(render_func("rgb_array")[:, :, ::-1])
        if frame_counter >= args.gif:
            if not os.path.isdir("./gifs"):
                os.mkdir("./gifs")
            now = datetime.now()
            gif_path = f"./gifs/{args.env_name}-{now.strftime('%y%m%d-%H%M%S')}.gif"
            imageio.mimsave(gif_path, images, fps=30, duration=1 / 10)
            print("===SAVED GIF TO THIS LOCATION: ", gif_path)
            quit()
