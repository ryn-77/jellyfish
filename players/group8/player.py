import random

from core import snapshot, current
from core.depth import Depth
from core.player import Player
from core.action import Action
from core.seasons import Season
from math import cos, radians, sin
from math import atan2, cos, sin, tau
from itertools import combinations, product
from core.snapshot import JellyfishSnapshot


class Player8(Player):
    """Group 8's jellyfish.

    Called once per living jellyfish per day. Read what the jellyfish senses
    from `snapshot`, update `snapshot.memory` in place if you want to
    remember something, and return an Action choosing today's depth.

    Do not keep per-jellyfish state on `self`: memory is the only memory.
    """

    def __init__(self, memory_size: int, lifetime_days: int, rng: random.Random) -> None:
        super().__init__(memory_size, lifetime_days, rng)
        self.current_speeds = [40, 20, 10, 5]
        self.first_k_days = 4
        self.mandatory_exploration_days = 30
        self.visited_flag = 1
        self.breeding_flag = 2
        self.feeding_flag = 4
        self.predator_flag = 8

    def act(self, snapshot: JellyfishSnapshot) -> Action:
        # Full grid needs 108 bytes; smaller budgets use a safe stateless fallback.
        if len(snapshot.memory) < 108:
            if snapshot.in_feeding_zone or snapshot.in_breeding_zone:
                return Action(Depth.DEEP)
            return Action(Depth.SURFACE)
        return self.act_as_per_highest_memory(snapshot)

    def act_as_per_highest_memory(self, s: JellyfishSnapshot) -> Action:
        """
        We have 1024 memory bytes | Value range	0–255
        Bytes 0–3: headings
        Bytes 4–7: position
        Bytes 8–107: observations for the 100 grid cells
        1 byte = 8 bits | 1-visited_flag 2-breeding_flag 3-feeding_flag 4-predator_flag
        Using total 107/1024 bytes
        """
        x, y = self.update_position(s)
        self.set_heading(s)
        self.update_grid_cell(s, x, y)

        # Unseen headings cannot be treated as real directions. Newborns
        # inherit learned headings, so they can skip this stage entirely.
        if any(s.memory[i] == 0 for i in range(4)):
            for depth in Depth:
                if s.memory[depth.value] == 0:
                    return Action(depth)

        # Exploration is now driven by whether the target is known, not
        # by a fixed 30-day timer. Initial breeding observations are retained.
        target_flag = (self.feeding_flag if s.season in (Season.SPRING, Season.SUMMER)
                       else self.breeding_flag)
        if not (s.in_feeding_zone if target_flag == self.feeding_flag else s.in_breeding_zone):
            if self.estimate_zone_center(s.memory, target_flag) is None:
                depth = self.explore_grid(s, x, y)
                if depth is None:
                    depth = Depth.SURFACE
                return Action(self.avoid_predator(s, x, y, depth))

        # Alternate between feeding and breeding
        if s.season == Season.SPRING:
            if s.in_feeding_zone:
                return self.stay_put(s, x, y, self.feeding_flag)
            else:
                return self.move_to_feeding_zone(s, x, y)
        if s.season == Season.SUMMER:
            if s.in_feeding_zone:
                return self.stay_put(s, x, y, self.feeding_flag)
            else:
                return self.move_to_feeding_zone(s, x, y)
        if s.season == Season.FALL:
            if s.in_breeding_zone:
                return self.stay_put(s, x, y, self.breeding_flag)
            else:
                return self.move_to_breeding_zone(s, x, y)
        if s.season == Season.WINTER:
            if s.in_breeding_zone:
                return self.stay_put(s, x, y, self.breeding_flag)
            else:
                return self.move_to_breeding_zone(s, x, y)

        return Action(random.choice(list(Depth)))
