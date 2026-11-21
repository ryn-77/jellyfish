from core.action import Action
from core.depth import Depth
from core.player import Player
from core.seasons import Season
from core.snapshot import JellyfishSnapshot


class Player1(Player):
    """Group 1: a simple baseline that uses no memory.

    A jellyfish can't see where it is, only which zones it is in, so each one
    follows four rules:

    1. In the predator zone, ride the surface current (40 km/day), the fastest
       way out.
    2. In the zone we need right now, ride the deep current (5 km/day) to stay
       in it as long as possible. In summer that is the feeding zone. In
       winter, if we can still reproduce, it is the breeding zone.
    3. If we need a zone and aren't in it, search on the surface current.
    4. Otherwise ride the deep current, so we barely move and rarely drift
       into the predator zone.
    """

    def act(self, snapshot: JellyfishSnapshot) -> Action:
        s = snapshot
        if s.in_predator_zone:
            return Action(Depth.SURFACE)

        if s.season == Season.SUMMER:
            in_target = s.in_feeding_zone
        elif s.season == Season.WINTER and s.reproduction_capacity > 0:
            in_target = s.in_breeding_zone
        else:
            return Action(Depth.DEEP)  # nothing to look for: stay put

        return Action(Depth.DEEP if in_target else Depth.SURFACE)
