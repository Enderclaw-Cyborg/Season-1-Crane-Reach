"""A working Crane starter agent.

Each unit runs a separate instance of this class. This starter walks forward until it sees an
enemy, then takes one legal step toward the nearest visible enemy and names it. Start at the
``TODO(you)`` comments.
Read ``environment.md`` beside this file for the rules, helpers, and first improvement. Prepare
episode state in ``reset``. The constructor takes no arguments.
"""

from sandbox.crane import action, me, paths, tile, visible, roster, units, zone
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation

class Agent:
    """Marches toward the enemy side, then steps toward the nearest visible enemy."""

    def reset(self, seed, observation) -> None:
        pass

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        # The enemies this unit can see.
        enemies = visible.enemies(observation)

        if not enemies:
            # With no enemy visible, head toward the mirrored position on the enemy side.
            # Searching all legal paths lets the unit route around walls instead of repeatedly
            # trying the blocked forward direction.
            here = me.position(observation)
            enemy_side_goal = tile.at_mirror(here, observation)
            path_id = self._path_toward(observation, enemy_side_goal)
            return action.move(path_id) if path_id else action.stay()

        # This unit's current {"q": ..., "r": ...} position.
        here = me.position(observation)

        # The closest enemy in sight. min returns the enemy dictionary, not the distance.
        nearest = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))

        #addition from documentation
        if me.unit_type(observation) == "archer":
            path_id = self._retreat_path(observation, nearest["position"])
        elif me.unit_type(observation) == "cavalry":
            path_id = self._flanking_path(observation, nearest["position"])
        else:
            path_id = self._path_toward(observation, nearest["position"])

        # Naming a target makes the strike prefer that enemy. Any visible enemy can be named,
        # so both orders below are legal.
        if path_id == 0:
            return action.stay(nearest["unit_id"], observation)
        return action.move(path_id, nearest["unit_id"], observation)

    def _path_toward(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Return the legal path that most closes the gap to goal, or 0 when none does."""
        here = me.position(observation)
        current_distance = tile.distance(here, goal)
        candidates: list[int] = []
        best_distance = current_distance

        for path_id in action.legal_paths(observation):
            if path_id == 0:
                continue

            path_distance = tile.distance(tile.at_path_end(here, path_id), goal)

            if path_distance < best_distance:
                candidates = [path_id]
                best_distance = path_distance
            elif path_distance == best_distance and path_distance < current_distance:
                candidates.append(path_id)

        if not candidates:
            return 0
        return candidates[0]

    def _retreat_path(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Return the legal path that maximizes distance from goal, or 0 when none does."""
        here = me.position(observation)
        candidates: list[int] = []
        best_distance = tile.distance(here, goal)

        for path_id in action.legal_paths(observation):
            if path_id == 0:
                continue

            path_distance = tile.distance(tile.at_path_end(here, path_id), goal)
            if path_distance > best_distance:
                candidates = [path_id]
                best_distance = path_distance
            elif path_distance == best_distance and path_distance > tile.distance(here, goal):
                candidates.append(path_id)

        return candidates[0] if candidates else 0

    def _flanking_path(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Return the longest legal path that still reduces the gap to goal, or 0."""
        here = me.position(observation)
        current_distance = tile.distance(here, goal)
        candidates: list[tuple[int, int, int]] = []

        for path_id in action.legal_paths(observation):
            if path_id == 0:
                continue

            path_distance = tile.distance(tile.at_path_end(here, path_id), goal)
            if path_distance < current_distance:
                candidates.append((-len(paths.decode(path_id)), path_distance, path_id))

        if not candidates:
            return 0
        longest_path = min(candidate[0] for candidate in candidates)
        longest = [candidate for candidate in candidates if candidate[0] == longest_path]
        return min(longest)[2]


    # Optional: a reinforcement-learning hook called after every step with that step's
    # transition. Its time counts against the timing and episode budget. The order argument is
    # what act returned. It is named order so it does not shadow the action helpers.
    #
    # def learn(self, observation, order: SkirmishAction, reward: float, terminated: bool) -> None:
    #     ...

    # Optional: messaging. Season settings enable it from Season 3 onward. When enabled, chat runs
    # after a unit chooses its order and receives messages that arrived since its previous
    # activation. Return each message with a recipient and text. Use None to broadcast to both
    # sides, or a player id such as "player_2", not a unit id, to send directly to one ally. The
    # rosters in the observation map each player to its unit. By default, text is limited to 200
    # characters.
    # A direct message reaches its allied unit at its next activation, after that unit chooses its
    # own order. Every message is recorded and shown in replays, so nothing you send is ever secret.
    # Return nothing to stay silent.
    #
    # def chat(self, inbox: list[dict]) -> list[dict] | None:
    #     ...
