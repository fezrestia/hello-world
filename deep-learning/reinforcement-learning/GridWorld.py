import numpy as np
from collections.abc import Iterator
import matplotlib
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self) -> None:
        self.action_space: list[int] = [0, 1, 2, 3]

        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map: np.ndarray = np.array(
                [
                    [0, 0,      0, 1.0],
                    [0, None,   0, -1.0],
                    [0, 0,      0, 0],
                ]
        )

        self.goal_state: tuple[int, int] = (0, 3)
        self.wall_state: tuple[int, int] = (1, 1)
        self.start_state: tuple[int, int] = (2, 0)
        self.agent_state: tuple[int, int] = self.start_state

    @property
    def height(self) -> int:
        return len(self.reward_map)

    @property
    def width(self) -> int:
        return len(self.reward_map[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.reward_map.shape

    def actions(self) -> list[int]:
        return self.action_space

    def states(self) -> Iterator[tuple[int, int]]:
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        action_move_map: list[tuple[int, int]] = [
                (-1, 0),  # UP
                (1, 0),  # DOWN
                (0, -1),  # LEFT
                (0, 1),  # RIGHT
        ]

        move: tuple[int, int] = action_move_map[action]

        next_state: tuple[int, int] = (state[0] + move[0], state[1] + move[1])
        (ny, nx) = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(
            self,
            state: tuple[int, int],
            action: int,
            next_state: tuple[int, int],
    ) -> float|None:
        return self.reward_map[next_state]

    def render_v(self, v = None, policy = None, print_value = True):
        renderer = Renderer(self.reward_map, self.goal_state, self.wall_state)
        renderer.render_v(v, policy, print_value)



class Renderer:
    def __init__(self, reward_map, goal_state, wall_state):
        self.reward_map = reward_map
        self.goal_state = goal_state
        self.wall_state = wall_state
        self.ys = len(self.reward_map)
        self.xs = len(self.reward_map[0])

        self.ax = None
        self.fig = None
        self.first_flg = True

    def set_figure(self, figsize=None):
        fig = plt.figure(figsize=figsize)
        self.ax = fig.add_subplot(111)
        ax = self.ax
        ax.clear()
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        ax.set_xticks(range(self.xs))
        ax.set_yticks(range(self.ys))
        ax.set_xlim(0, self.xs)
        ax.set_ylim(0, self.ys)
        ax.grid(True)

    def render_v(self, v=None, policy=None, print_value=True):
        self.set_figure()

        ys, xs = self.ys, self.xs
        ax = self.ax

        if v is not None:
            color_list = ['red', 'white', 'green']
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'colormap_name', color_list)

            # dict -> ndarray
            v_dict = v
            v = np.zeros(self.reward_map.shape)
            for state, value in v_dict.items():
                v[state] = value

            vmax, vmin = v.max(), v.min()
            vmax = max(vmax, abs(vmin))
            vmin = -1 * vmax
            vmax = 1 if vmax < 1 else vmax
            vmin = -1 if vmin > -1 else vmin

            ax.pcolormesh(np.flipud(v), cmap=cmap, vmin=vmin, vmax=vmax)

        for y in range(ys):
            for x in range(xs):
                state = (y, x)
                r = self.reward_map[y, x]
                if r != 0 and r is not None:
                    txt = 'R ' + str(r)
                    if state == self.goal_state:
                        txt = txt + ' (GOAL)'
                    ax.text(x+.1, ys-y-0.9, txt)

                if (v is not None) and state != self.wall_state:
                    if print_value:
                        offsets = [(0.4, -0.15), (-0.15, -0.3)]
                        key = 0
                        if v.shape[0] > 7: key = 1
                        offset = offsets[key]
                        ax.text(x+offset[0], ys-y+offset[1], "{:12.2f}".format(v[y, x]))

                if policy is not None and state != self.wall_state:
                    actions = policy[state]
                    max_actions = [kv[0] for kv in actions.items() if kv[1] == max(actions.values())]

                    arrows = ["↑", "↓", "←", "→"]
                    offsets = [(0, 0.1), (0, -0.1), (-0.1, 0), (0.1, 0)]
                    for action in max_actions:
                        arrow = arrows[action]
                        offset = offsets[action]
                        if state == self.goal_state:
                            continue
                        ax.text(x+0.45+offset[0], ys-y-0.5+offset[1], arrow)

                if state == self.wall_state:
                    ax.add_patch(plt.Rectangle((x,ys-y-1), 1, 1, fc=(0.4, 0.4, 0.4, 1.)))
        plt.show()

