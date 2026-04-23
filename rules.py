"""
Rule based engine for Street Fighter II.

Evaluates the current GameState and returns:
  strategy  : "aggressive" | "defensive" | "neutral"
  override  : a concrete 12 button action array, or None to defer to PPO
  reward_adj: an additive reward shaping term

Rules are evaluated top to bottom; the first rule whose condition fires
produces the override (if any).  Strategy and reward shaping accumulate
from all matching rules.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from game_state import GameState


@dataclass
class RuleOutput:
    """Result returned by the rule engine each step."""
    strategy: str = "neutral"                       
    override: Optional[np.ndarray] = None           
    reward_adjustment: float = 0.0                  
    rule_name: str = ""                             


def _action(*buttons) -> np.ndarray:
    """Create a 12 dimensional binary action with the given buttons pressed."""
    a = np.zeros(12, dtype=np.int8)
    for b in buttons:
        a[b] = 1
    return a

GS = GameState  

ACTION_BLOCK_STANDING = _action(GS.BTN_LEFT)           
ACTION_BLOCK_CROUCHING = _action(GS.BTN_LEFT, GS.BTN_DOWN)
ACTION_LIGHT_PUNCH     = _action(GS.BTN_Y)
ACTION_HEAVY_PUNCH     = _action(GS.BTN_Z)
ACTION_HEAVY_KICK      = _action(GS.BTN_C)
ACTION_CROUCH_KICK     = _action(GS.BTN_DOWN, GS.BTN_B)
ACTION_JUMP_KICK       = _action(GS.BTN_UP, GS.BTN_C)
ACTION_FORWARD         = _action(GS.BTN_RIGHT)
ACTION_IDLE            = _action()

SPECIAL_HADOUKEN = [
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_RIGHT, GS.BTN_Y),               
    _action(GS.BTN_RIGHT, GS.BTN_Y),               
    _action(),                                     
]

SPECIAL_SHORYUKEN = [
    _action(GS.BTN_RIGHT),                         
    _action(GS.BTN_RIGHT),                         
    _action(GS.BTN_RIGHT),                         
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN, GS.BTN_RIGHT, GS.BTN_Z), 
    _action(GS.BTN_DOWN, GS.BTN_RIGHT, GS.BTN_Z), 
    _action(),                                     
]

SPECIAL_TATSUMAKI = [
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN, GS.BTN_LEFT),             
    _action(GS.BTN_DOWN, GS.BTN_LEFT),             
    _action(GS.BTN_DOWN, GS.BTN_LEFT),             
    _action(GS.BTN_LEFT, GS.BTN_C),                
    _action(GS.BTN_LEFT, GS.BTN_C),                
    _action(),                                     
]

COMBO_PRESSURE = [
    _action(GS.BTN_Y),                            
    _action(GS.BTN_Y),                            
    _action(GS.BTN_DOWN, GS.BTN_B),               
    _action(GS.BTN_Z),                            
]

COMBO_PUNISH = [
    _action(GS.BTN_Z),                            
    _action(GS.BTN_C),                            
]

COMBO_CR_MK_HADOUKEN = [
    _action(GS.BTN_DOWN, GS.BTN_A),               
    _action(GS.BTN_DOWN, GS.BTN_A),               
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_RIGHT, GS.BTN_Y),               
    _action(GS.BTN_RIGHT, GS.BTN_Y),               
    _action(),                                     
]

COMBO_HP_SHORYUKEN = [
    _action(GS.BTN_Z),                            
    _action(GS.BTN_Z),                            
    _action(GS.BTN_RIGHT),                         
    _action(GS.BTN_RIGHT),                         
    _action(GS.BTN_RIGHT),                         
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN, GS.BTN_RIGHT, GS.BTN_Z), 
    _action(GS.BTN_DOWN, GS.BTN_RIGHT, GS.BTN_Z), 
    _action(),                                     
]

COMBO_FULL_CONFIRM = [
    _action(GS.BTN_Y),                            
    _action(GS.BTN_Y),                            
    _action(GS.BTN_DOWN, GS.BTN_A),               
    _action(GS.BTN_DOWN, GS.BTN_A),               
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_RIGHT, GS.BTN_Y),               
    _action(GS.BTN_RIGHT, GS.BTN_Y),               
    _action(),                                     
]

COMBO_JUMPIN = [
    _action(GS.BTN_UP, GS.BTN_RIGHT, GS.BTN_C),  
    _action(GS.BTN_UP, GS.BTN_RIGHT, GS.BTN_C),  
    _action(),                                     
    _action(),                                     
    _action(),                                     
    _action(GS.BTN_DOWN, GS.BTN_Z),               
    _action(GS.BTN_DOWN, GS.BTN_Z),               
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_RIGHT, GS.BTN_Y),               
    _action(GS.BTN_RIGHT, GS.BTN_Y),               
    _action(),                                     
]

COMBO_CLOSE_HP_CR_HP_HADOUKEN = [
    _action(GS.BTN_Z),                             
    _action(GS.BTN_Z),                             
    _action(GS.BTN_DOWN, GS.BTN_Z),                
    _action(GS.BTN_DOWN, GS.BTN_Z),                
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            
    _action(GS.BTN_RIGHT, GS.BTN_Y),               
    _action(GS.BTN_RIGHT, GS.BTN_Y),               
    _action(),                                     
]

COMBO_JUMPIN_HP_SHORYUKEN = [
    _action(GS.BTN_UP, GS.BTN_RIGHT, GS.BTN_C),   
    _action(GS.BTN_UP, GS.BTN_RIGHT, GS.BTN_C),   
    _action(),                                     
    _action(),                                     
    _action(),                                     
    _action(GS.BTN_Z),                             
    _action(GS.BTN_Z),                             
    _action(GS.BTN_RIGHT),                         
    _action(GS.BTN_RIGHT),                         
    _action(GS.BTN_RIGHT),                         
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN),                          
    _action(GS.BTN_DOWN, GS.BTN_RIGHT, GS.BTN_X), 
    _action(GS.BTN_DOWN, GS.BTN_RIGHT, GS.BTN_X), 
    _action(),                                     
]


class RuleEngine:
    """Stateful rule engine call evaluate() once per step."""

    def __init__(self):
        self._combo_queue: list[np.ndarray] = []
        self._block_frames: int = 0
        self._idle_penalised: bool = False   
        self._hadouken_cooldown: int = 0     

    def reset(self):
        self._combo_queue.clear()
        self._block_frames = 0
        self._idle_penalised = False
        self._hadouken_cooldown = 0

    def evaluate(self, state: GameState) -> RuleOutput:
        out = RuleOutput()

        if state.damage_dealt > 0:
            self._idle_penalised = False

        if self._hadouken_cooldown > 0:
            self._hadouken_cooldown -= 1

        if self._combo_queue:
            out.override = self._combo_queue.pop(0)
            out.rule_name = "combo_continuation"
            return out

        if self._block_frames > 0:
            self._block_frames -= 1
            out.override = ACTION_BLOCK_STANDING.copy()
            out.strategy = "defensive"
            out.rule_name = "block_continuation"
            return out

        if state.damage_taken > 30:
            self._block_frames = 8
            out.override = ACTION_BLOCK_CROUCHING.copy()
            out.strategy = "defensive"
            out.rule_name = "emergency_block"
            out.reward_adjustment += 3.0
            return out

        if state.blocked_hit:
            out.reward_adjustment += 2.0
            out.strategy = "defensive"
            out.rule_name = "block_reward"

        if state.successful_dodge:
            out.reward_adjustment += 3.0
            out.strategy = "aggressive"
            out.rule_name = "dodge_reward"

        if (state.player_low_health(0.20)
                and not state.enemy_low_health(0.20)
                and state.damage_taken > 0):
            out.strategy = "defensive"
            out.reward_adjustment += 3.0
            self._block_frames = 8
            out.override = ACTION_BLOCK_STANDING.copy()
            out.rule_name = "low_health_block"
            return out

        if (state.enemy_low_health(0.20)
                and state.damage_dealt > 0
                and not self._combo_queue):
            out.strategy = "aggressive"
            out.reward_adjustment += 5.0
            self._combo_queue = [a.copy() for a in COMBO_HP_SHORYUKEN]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "finish_shoryuken"
            return out

        if state.damage_dealt > 25 and state.steps_since_damage_dealt == 0:
            out.strategy = "aggressive"
            out.reward_adjustment += 4.0
            self._combo_queue = [a.copy() for a in COMBO_CLOSE_HP_CR_HP_HADOUKEN]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "heavy_confirm"
            return out

        if state.damage_dealt > 20 and state.steps_since_damage_dealt == 0:
            out.strategy = "aggressive"
            out.reward_adjustment += 3.0
            self._combo_queue = [a.copy() for a in COMBO_CR_MK_HADOUKEN]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "mid_confirm"
            return out

        if state.damage_dealt > 10 and state.steps_since_damage_dealt == 0:
            out.strategy = "aggressive"
            out.reward_adjustment += 2.0
            self._combo_queue = [a.copy() for a in COMBO_PRESSURE]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "light_confirm"
            return out

        if (state.health_advantage > 0.15
                and not state.enemy_low_health(0.20)
                and not state.player_low_health(0.30)
                and state.steps_since_damage_dealt > 30
                and state.steps_since_damage_taken > 20
                and not self._combo_queue):
            out.strategy = "aggressive"
            out.reward_adjustment += 3.0
            self._combo_queue = [a.copy() for a in COMBO_JUMPIN_HP_SHORYUKEN]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "jumpin_hp_shoryuken"
            return out

        if (20 < state.steps_since_damage_dealt <= 45
                and state.steps_since_damage_taken > 10
                and not state.player_low_health(0.20)
                and not self._combo_queue):
            out.strategy = "aggressive"
            out.reward_adjustment += 1.0
            self._combo_queue = [a.copy() for a in COMBO_PRESSURE]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "proactive_pressure"
            return out

        if (45 < state.steps_since_damage_dealt <= 60
                and state.steps_since_damage_taken > 15
                and not state.player_low_health(0.20)
                and not self._combo_queue):
            out.strategy = "aggressive"
            out.reward_adjustment += 0.5
            self._combo_queue = [
                _action(GS.BTN_RIGHT),               
                _action(GS.BTN_RIGHT),               
                _action(GS.BTN_RIGHT, GS.BTN_Y),     
            ]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "approach_jab"
            return out

        if (state.steps_since_damage_dealt > 60
                and state.steps_since_damage_taken > 60
                and not state.player_low_health(0.20)
                and self._hadouken_cooldown == 0):
            out.strategy = "aggressive"
            out.reward_adjustment += 0.5
            self._combo_queue = [a.copy() for a in SPECIAL_HADOUKEN]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "zoning_hadouken"
            self._hadouken_cooldown = 45
            return out

        if state.is_idle(patience=200) and not self._idle_penalised:
            out.strategy = "aggressive"
            out.reward_adjustment += -5.0
            out.rule_name = "anti_idle"
            self._idle_penalised = True
            return out

        out.strategy = "neutral"
        out.rule_name = "defer_to_ppo"
        return out