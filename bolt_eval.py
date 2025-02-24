import argparse
import os
import pickle

import torch
from bolt_env import boltEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

import pygame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="bolt-walking")
    parser.add_argument("--ckpt", type=int, default=1000)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = boltEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            keys = pygame.key.get_pressed()
            commands = torch.zeros(3)
            if keys[pygame.K_UP]:
                commands[0] = 1
            if keys[pygame.K_DOWN]:
                commands[0] = -1
            if keys[pygame.K_LEFT]:
                commands[1] = -1
            if keys[pygame.K_RIGHT]:
                commands[1] = 1
            if keys[pygame.K_Q]:
                commands[2] = 1
            if keys[pygame.K_D]:
                commands[2] = -1
                
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
