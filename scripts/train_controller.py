"""Train the controller with CMA-ES in the real CarRacing env.

Rollout loop (per step):
    obs -> VAE.encode -> z
    action = Controller(z, h)     # h = MDN-RNN hidden state
    env.step(action)
    h = MDN-RNN(z, a, h).hidden   # update memory

Fitness = mean episode return across K rollouts. CMA-ES maximizes this.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cma
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from world_model.controller import Controller  # noqa: E402
from world_model.mdn_rnn import MDNRNN, MDNRNNConfig  # noqa: E402
from world_model.vae import VAE, VAEConfig  # noqa: E402


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_vae(path: Path, device: torch.device) -> VAE:
    blob = torch.load(path, map_location=device, weights_only=False)
    m = VAE(VAEConfig(**blob["config"])).to(device).eval()
    m.load_state_dict(blob["model"])
    return m


def load_mdn_rnn(path: Path, device: torch.device) -> MDNRNN:
    blob = torch.load(path, map_location=device, weights_only=False)
    m = MDNRNN(MDNRNNConfig(**blob["config"])).to(device).eval()
    m.load_state_dict(blob["model"])
    return m


def preprocess_obs(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """96x96x3 uint8 -> (1, 3, 64, 64) float [0, 1]."""
    img = Image.fromarray(obs).resize((64, 64), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def rollout(
    env: gym.Env,
    vae: VAE,
    mdn: MDNRNN,
    controller: Controller,
    device: torch.device,
    max_steps: int,
    action_repeat: int,
    seed: int,
    render_frames: list | None = None,
) -> float:
    obs, _ = env.reset(seed=seed)
    h = torch.zeros(1, mdn.cfg.hidden_dim, device=device)
    c_state = torch.zeros(1, mdn.cfg.hidden_dim, device=device)

    total_reward = 0.0
    steps = 0
    while steps < max_steps:
        x = preprocess_obs(obs, device)
        mu, _ = vae.encode(x)             # (1, 32)
        z = mu

        action_t = controller(z, h)        # (1, 3)
        action = action_t.cpu().numpy().squeeze(0)

        # Update LSTM state with (z, a)
        z_in = z.unsqueeze(1)             # (1, 1, 32)
        a_in = action_t.unsqueeze(1)      # (1, 1, 3)
        _, _, _, _, (h_new, c_new) = mdn(z_in, a_in, (h.unsqueeze(0), c_state.unsqueeze(0)))
        h = h_new.squeeze(0)
        c_state = c_new.squeeze(0)

        # action repeat in the real env
        for _ in range(action_repeat):
            if render_frames is not None:
                render_frames.append(env.render())
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated or steps >= max_steps:
                break
        if terminated or truncated:
            break

    return total_reward


def evaluate(params: np.ndarray, context: dict, n_rollouts: int, seed: int) -> float:
    """Load params into the controller, run n_rollouts, return mean return."""
    ctrl: Controller = context["controller"]
    ctrl.set_flat_params(params)
    returns = []
    for i in range(n_rollouts):
        r = rollout(
            context["env"],
            context["vae"],
            context["mdn"],
            ctrl,
            context["device"],
            max_steps=context["max_steps"],
            action_repeat=context["action_repeat"],
            seed=seed + i,
        )
        returns.append(r)
    return float(np.mean(returns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", type=Path, default=Path("checkpoints/vae.pt"))
    parser.add_argument("--mdn-ckpt", type=Path, default=Path("checkpoints/mdn_rnn.pt"))
    parser.add_argument("--ctrl-ckpt", type=Path, default=Path("checkpoints/controller.pt"))
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--sigma0", type=float, default=0.1)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    vae = load_vae(args.vae_ckpt, device)
    mdn = load_mdn_rnn(args.mdn_ckpt, device)
    controller = Controller().to(device)
    print(f"Controller params: {controller.num_params}")

    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    context = {
        "env": env, "vae": vae, "mdn": mdn, "controller": controller,
        "device": device, "max_steps": args.max_steps,
        "action_repeat": args.action_repeat,
    }

    x0 = controller.get_flat_params()
    es = cma.CMAEvolutionStrategy(
        x0, args.sigma0,
        {"popsize": args.popsize, "seed": args.seed + 1, "verbose": -9},
    )

    args.ctrl_ckpt.parent.mkdir(parents=True, exist_ok=True)
    best_return = -1e9
    best_params = x0

    for gen in range(1, args.generations + 1):
        t0 = time.time()
        candidates = es.ask()
        fitnesses = []
        returns = []
        for i, params in enumerate(candidates):
            ret = evaluate(params, context, args.rollouts, seed=args.seed + gen * 100 + i)
            returns.append(ret)
            fitnesses.append(-ret)  # CMA-ES minimizes
        es.tell(candidates, fitnesses)

        best_idx = int(np.argmax(returns))
        if returns[best_idx] > best_return:
            best_return = returns[best_idx]
            best_params = candidates[best_idx]
            controller.set_flat_params(best_params)
            torch.save(
                {
                    "state_dict": controller.state_dict(),
                    "flat_params": best_params,
                    "return": best_return,
                    "generation": gen,
                },
                args.ctrl_ckpt,
            )

        dt = time.time() - t0
        print(
            f"gen {gen:2d} | mean={np.mean(returns):7.2f} max={max(returns):7.2f} "
            f"best_so_far={best_return:7.2f} ({dt:.1f}s)"
        )

    env.close()
    print(f"\nBest return: {best_return:.2f}")
    print(f"Saved {args.ctrl_ckpt}")


if __name__ == "__main__":
    main()
