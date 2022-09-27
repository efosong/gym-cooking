from gym_cooking.environment.game.game import Game
from gym_cooking.environment import cooking_zoo
import os
import torch
import numpy as np
import tempfile
from subprocess import DEVNULL, STDOUT, check_call
from d_network import AgentNetworkFC
import hydra

class RandomAgent:

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation) -> int:
        return self.action_space.sample()

class DeepAgent:

    def __init__(self, model_path):
        model = torch.load(model_path)
        self.joint_policy = AgentNetworkFC(
            obs_dim = model["linear_obs_rep.0.weight"].shape[1],
            obs_u_dim = 0,
            mid_dim1 = model["linear_obs_rep.2.weight"].shape[0],
            mid_dim2 = model["linear_obs_rep.2.weight"].shape[1],
            gnn_hdim1 = model["gnn.2.weight"].shape[0],
            gnn_hdim2 = model["gnn.2.weight"].shape[1],
            gnn_out_dim = model["gnn.4.weight"].shape[0],
            num_acts = model["act_logits.weight"].shape[0],
            device="cpu",
        )
        self.joint_policy.load_state_dict(model)

    def get_action(self, observation):
        logits = self.joint_policy(torch.Tensor(observation))
        dist = torch.distributions.categorical.Categorical(logits=logits)
        act = dist.sample().item()
        return act


@hydra.main(version_base=None, config_path="config", config_name="record")
def run_game(cfg):
    video_save_dir = ""
    video_save_path = os.path.join(video_save_dir, cfg.viz.video_save_name)

    parallel_env = cooking_zoo.parallel_env(
        level=cfg.env.level,
        num_agents=cfg.env.n_agents,
        max_steps=cfg.env.max_steps,
        recipes=cfg.env.recipes,
        obs_spaces=cfg.env.obs_spaces,
        record=cfg.viz.record,
        )
    a1 = DeepAgent(cfg.agents.agent_1.path)
    a2 = DeepAgent(cfg.agents.agent_2.path)
    agents = [a1, a2]
    delay = 1/cfg.viz.fps if cfg.viz.render else 0

    with tempfile.TemporaryDirectory() as temp_dir:
        game = Game(
            env=parallel_env,
            num_humans=0,
            ai_policies=agents,
            render=cfg.viz.render,
            record=cfg.viz.record,
            save_dir=temp_dir,
            )
        store = game.on_execute_ai_only_with_delay(delay=delay)
        if cfg.viz.record:
            img_location = os.path.join(temp_dir, r"output_%03d.png")
            check_call(
                ["ffmpeg", "-r", str(cfg.viz.fps), "-f", "image2", "-i", img_location, "-y", video_save_path],
                stdout=DEVNULL,
                stderr=STDOUT
                )

    if cfg.viz.record:
        print(f"Saved video to {video_save_path}")

if __name__ == "__main__":
    run_game()
