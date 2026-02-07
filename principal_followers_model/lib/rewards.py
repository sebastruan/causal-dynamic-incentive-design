def place_reward(occupancy: int, capacity: int,
                 r_match: float = 10.0,
                 r_under: float = 1.0,
                 r_over: float = -5.0) -> float:
    """
    Piecewise reward:
      - exactly capacity -> high reward
      - below capacity -> low reward (scaled by occupancy/capacity)
      - above capacity -> punishment that grows with overflow
    """
    if capacity <= 0:
        return 0.0

    if occupancy == capacity:
        return r_match
    elif occupancy < capacity:
        # low reward; increases as it approaches capacity
        return r_under * (occupancy / capacity)
    else:
        # punishment; stronger as overflow increases
        overflow = occupancy - capacity
        return r_over * (1.0 + overflow / capacity)
