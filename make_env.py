
from src.env import GridWorldEnv

def make_gridworld_env(
    n_rows=5,
    n_cols=5,
    start_pos=(1, 0),
    terminal_pos=(4, 4),
    jump_from=(1, 3),
    jump_to=(3, 3),
    obstacles=None,
):
    """
    Instantiate and return a GridWorldEnv.
    Parameters can be customized for different grid testing.
    """
    if obstacles is None:
        # Default to assignment obstacles; adjust as needed!
        obstacles = [(2, 1), (2, 2)]
    env = GridWorldEnv(
        n_rows=n_rows,
        n_cols=n_cols,
        start_pos=start_pos,
        terminal_pos=terminal_pos,
        jump_from=jump_from,
        jump_to=jump_to,
        obstacles=obstacles
    )
    return env

# Example usage for illustration / debugging
if __name__ == "__main__":
    env = make_gridworld_env()
    state = env.reset()
    print("Initial state:", state)
    print("Environment created! Ready for agent interaction.")