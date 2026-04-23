"""
Baseline (PPO-only) Street Fighter II environment.

Pure reinforcement learning — no rule engine, no gate, no game features.
Uses the same reward shaping as the adaptive env (minus rule adjustments)
so results are directly comparable.

Architecture
------------
  Action = MultiBinary(12)     — standard Genesis controller buttons
  Observation = (84, 84, 4)    — 4 stacked grayscale frames

This serves as the control group for evaluating how much the
rule engine + learned gating improves performance.
"""

from collections import deque

import gymnasium as gym
from gymnasium.spaces import MultiBinary, Box

import numpy as np
import cv2
import pygame
import retro

from game_state import GameState

NATIVE_W, NATIVE_H = 320, 224
SCALE = 3
N_FRAME_STACK = 4


class BaselineStreetFighter(gym.Env):
    """SF2 env with pure PPO — no rule engine."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, reward_shaping=True):
        super().__init__()

        # --- Observation space ---
        # 4 stacked grayscale 84x84 frames (same as adaptive)
        self.observation_space = Box(low=0, high=255,
                                     shape=(84, 84, N_FRAME_STACK),
                                     dtype=np.uint8)

        # --- Action space ---
        # 12 game buttons only — no gate bit
        self.action_space = MultiBinary(12)

        self.game = retro.make(
            game="StreetFighterIISpecialChampionEdition-Genesis",
            use_restricted_actions=retro.Actions.FILTERED,
        )

        # Internal frame stack (same as adaptive)
        self._frames = deque(maxlen=N_FRAME_STACK)

        # Rendering state
        self.rendering = (render_mode == "human")
        self._screen = None
        self._render_w = None
        self._render_h = None

        # Game state tracker (for reward shaping only — no rule engine)
        self.game_state = GameState()
        self.reward_shaping = reward_shaping

        # Reward-shaping weights — IDENTICAL to adaptive env
        # (minus w_rule_adj since there is no rule engine)
        self.w_damage_dealt = 0.3
        self.w_damage_taken = -1.2
        self.w_win_round = 400.0
        self.w_lose_round = -300.0
        self.w_block = 0.5
        self.w_dodge = 1.5
        self.w_combo = 1.0
        self.w_combo_cap = 5
        self.w_health_preservation = 50.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs = self.game.reset()
        frame = self._preprocess(raw_obs)

        # Fill frame stack with copies of the first frame
        self._frames.clear()
        for _ in range(N_FRAME_STACK):
            self._frames.append(frame)

        self.score = 0
        self.game_state.reset()
        self._prev_matches_won = 0
        self._prev_enemy_matches_won = 0

        return self._stacked_pixels(), {}

    def step(self, action):
        action = np.asarray(action).flatten()

        # Step the game directly — no gate, no rule engine
        raw_obs, _reward, done, info = self.game.step(action)

        if self.rendering:
            self._render_frame(raw_obs)

        # Preprocess and update frame stack
        frame = self._preprocess(raw_obs)
        self._frames.append(frame)

        # Update game state from RAM
        self.game_state.update(info)

        # Compute reward
        if self.reward_shaping:
            reward = self._shaped_reward(info)
        else:
            reward = info["score"] - self.score

        self.score = info["score"]

        # Info dict for logging
        info["game_state"] = self.game_state.as_dict()

        return self._stacked_pixels(), reward, done, False, info

    def close(self):
        if self._screen is not None:
            pygame.quit()
        self.game.close()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _stacked_pixels(self):
        """Concatenate the frame stack along the channel axis -> (84,84,4)."""
        return np.concatenate(list(self._frames), axis=-1)

    # ------------------------------------------------------------------
    # Reward shaping — same as adaptive, without rule engine adjustment
    # ------------------------------------------------------------------
    def _shaped_reward(self, info):
        gs = self.game_state
        reward = 0.0

        # 1. Damage-based reward
        reward += self.w_damage_dealt * gs.damage_dealt
        reward += self.w_damage_taken * gs.damage_taken

        # 2. Round outcome bonuses
        matches_won = info.get("matches_won", 0)
        enemy_matches_won = info.get("enemy_matches_won", 0)

        if matches_won > self._prev_matches_won:
            reward += self.w_win_round
            if gs.health_ratio > 0.5:
                reward += self.w_health_preservation
        if enemy_matches_won > self._prev_enemy_matches_won:
            reward += self.w_lose_round

        self._prev_matches_won = matches_won
        self._prev_enemy_matches_won = enemy_matches_won

        # 3. Block / dodge bonuses
        if gs.blocked_hit:
            reward += self.w_block
        if gs.successful_dodge:
            reward += self.w_dodge

        # 4. Combo bonus — capped
        if gs.combo_count >= 2:
            reward += self.w_combo * min(gs.combo_count, self.w_combo_cap)

        return reward

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(observation):
        """Convert raw RGB frame to grayscale 84x84x1."""
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resized, (84, 84, 1))

    def _render_frame(self, obs):
        if self._screen is None:
            pygame.init()
            actual_h, actual_w, _ = obs.shape
            self._render_w = actual_w * SCALE
            self._render_h = actual_h * SCALE
            self._screen = pygame.display.set_mode(
                (self._render_w, self._render_h)
            )
            pygame.display.set_caption("Street Fighter II — Baseline (PPO Only)")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        surface = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surface,
                                        (self._render_w, self._render_h))
        self._screen.blit(scaled, (0, 0))
        pygame.display.flip()
