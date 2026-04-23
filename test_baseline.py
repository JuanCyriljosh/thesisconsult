"""
Test script for the baseline (PPO-only) Street Fighter II agent.

Usage:
    python test_baseline.py
    python test_baseline.py --model ./opt_baseline/baseline_final_model --episodes 3
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from baseline_env import BaselineStreetFighter


def test(model_path, num_episodes=1):
    model = PPO.load(model_path)

    env = BaselineStreetFighter(render_mode="human", reward_shaping=True)
    env = DummyVecEnv([lambda: env])

    print("\nMode: BASELINE (PPO Only)")
    print("Model: {}".format(model_path))
    print("Episodes: {}\n".format(num_episodes))

    total_wins = 0
    total_losses = 0

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        prev_matches_won = 0
        prev_enemy_matches_won = 0

        print("=== Episode {} ===".format(episode))

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, vec_info = env.step(action)

            info = vec_info[0]
            step_count += 1
            total_reward += reward[0]

            # Track round outcomes
            gs = info.get("game_state", {})
            mw = gs.get("matches_won", 0)
            emw = gs.get("enemy_matches_won", 0)

            if mw > prev_matches_won:
                total_wins += 1
                print("  Step {:5d} | WIN  | R={:+7.1f} | Total={:8.1f} | "
                      "HP={}/{}".format(
                          step_count, reward[0], total_reward,
                          gs.get("health", "?"), gs.get("enemy_health", "?")))
            elif emw > prev_enemy_matches_won:
                total_losses += 1
                print("  Step {:5d} | LOSS | R={:+7.1f} | Total={:8.1f} | "
                      "HP={}/{}".format(
                          step_count, reward[0], total_reward,
                          gs.get("health", "?"), gs.get("enemy_health", "?")))
            elif reward[0] != 0:
                print("  Step {:5d} |      | R={:+7.1f} | Total={:8.1f} | "
                      "HP={}/{}".format(
                          step_count, reward[0], total_reward,
                          gs.get("health", "?"), gs.get("enemy_health", "?")))

            prev_matches_won = mw
            prev_enemy_matches_won = emw

        print("--- Episode {} done ---".format(episode))
        print("  Steps: {}  Reward: {:.1f}".format(step_count, total_reward))
        print()

    print("=" * 50)
    print("BASELINE SUMMARY")
    print("  Total episodes: {}".format(num_episodes))
    print("  Rounds won:     {}".format(total_wins))
    print("  Rounds lost:    {}".format(total_losses))
    if total_wins + total_losses > 0:
        print("  Win rate:       {:.1f}%".format(
            100 * total_wins / (total_wins + total_losses)))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test baseline (PPO-only) SF2 agent")
    parser.add_argument("--model", type=str,
                        default="./opt_baseline/baseline_final_model",
                        help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    args = parser.parse_args()

    test(model_path=args.model, num_episodes=args.episodes)
