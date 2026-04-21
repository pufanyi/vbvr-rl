"""Self-contained maze layout + video renderer for synthetic RL data.

Produces:
    - a perfect maze (DFS backtracker) on an ``(2h+1, 2w+1)`` grid
      (``1 = wall``, ``0 = passage``);
    - the BFS shortest path from start to goal;
    - a sequence of RGB frames showing a ball travelling that path at a
      configurable number of frames;
    - all metadata needed by a custom reward (grid, start, goal, path,
      per-frame ball position, palette, pixel geometry).

No torch / CUDA dependencies — this module is pure Python + numpy + PIL so it
can be imported by the precompute pipeline, the reward implementation, and
offline visualisation tools.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------


class MazePalette(BaseModel):
    """Named RGB palette used to render one maze sample.

    Colors are 0-255 uint8 triplets.  Names are only used to build the text
    prompt and to record which palette was sampled.
    """

    wall_rgb: tuple[int, int, int]
    passage_rgb: tuple[int, int, int]
    ball_rgb: tuple[int, int, int]
    goal_rgb: tuple[int, int, int]
    wall_name: str
    passage_name: str
    ball_name: str
    goal_name: str


DEFAULT_PALETTES: list[MazePalette] = [
    MazePalette(
        wall_rgb=(20, 20, 20),
        passage_rgb=(240, 240, 240),
        ball_rgb=(220, 40, 40),
        goal_rgb=(40, 180, 80),
        wall_name="black",
        passage_name="white",
        ball_name="red",
        goal_name="green",
    ),
    MazePalette(
        wall_rgb=(30, 50, 90),
        passage_rgb=(235, 230, 210),
        ball_rgb=(240, 170, 30),
        goal_rgb=(180, 60, 60),
        wall_name="navy",
        passage_name="cream",
        ball_name="orange",
        goal_name="crimson",
    ),
    MazePalette(
        wall_rgb=(50, 50, 50),
        passage_rgb=(210, 220, 235),
        ball_rgb=(60, 120, 230),
        goal_rgb=(240, 200, 60),
        wall_name="slate",
        passage_name="ice",
        ball_name="blue",
        goal_name="gold",
    ),
    MazePalette(
        wall_rgb=(60, 35, 20),
        passage_rgb=(245, 235, 215),
        ball_rgb=(200, 60, 160),
        goal_rgb=(70, 170, 110),
        wall_name="brown",
        passage_name="parchment",
        ball_name="magenta",
        goal_name="emerald",
    ),
]


# ---------------------------------------------------------------------------
# Maze layout + path
# ---------------------------------------------------------------------------


def generate_maze_grid(cell_h: int, cell_w: int, rng: np.random.Generator) -> np.ndarray:
    """Recursive-backtracker perfect maze.

    Cells live at odd grid coordinates, walls at even coordinates.  The
    returned array has shape ``(2*cell_h + 1, 2*cell_w + 1)`` and dtype uint8
    (``1 = wall``, ``0 = passage``).
    """
    H, W = 2 * cell_h + 1, 2 * cell_w + 1
    grid = np.ones((H, W), dtype=np.uint8)

    visited = np.zeros((cell_h, cell_w), dtype=bool)
    stack: list[tuple[int, int]] = [(0, 0)]
    visited[0, 0] = True
    grid[1, 1] = 0

    while stack:
        ci, cj = stack[-1]
        # Unvisited neighbours in random order.
        neighbours: list[tuple[int, int, int, int]] = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < cell_h and 0 <= nj < cell_w and not visited[ni, nj]:
                neighbours.append((ni, nj, di, dj))
        if not neighbours:
            stack.pop()
            continue
        pick = int(rng.integers(0, len(neighbours)))
        ni, nj, di, dj = neighbours[pick]
        # Carve the wall between (ci, cj) and (ni, nj) as well as the cell.
        grid[2 * ci + 1 + di, 2 * cj + 1 + dj] = 0
        grid[2 * ni + 1, 2 * nj + 1] = 0
        visited[ni, nj] = True
        stack.append((ni, nj))

    return grid


def bfs_shortest_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """4-neighbour BFS on passage cells. ``start``/``goal`` are grid coords."""
    H, W = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        raise ValueError("start or goal falls on a wall cell")

    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        ci, cj = cur
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == 0 and (ni, nj) not in parents:
                parents[(ni, nj)] = cur
                q.append((ni, nj))

    if goal not in parents:
        raise RuntimeError("BFS could not reach goal — grid is not a perfect maze")

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell_center_xy(i: int | float, j: int | float, cell_px: int) -> tuple[float, float]:
    """Pixel (x, y) of the centre of grid cell (i, j)."""
    return (j * cell_px + cell_px / 2.0, i * cell_px + cell_px / 2.0)


def interpolate_ball_positions(
    path: list[tuple[int, int]],
    num_frames: int,
    cell_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate the ball along ``path`` over ``num_frames`` frames.

    Returns ``(frame_positions_cell, frame_positions_pix)``, each shape
    ``(num_frames, 2)`` as float32.  ``cell`` coordinates are (i, j) row/col,
    ``pix`` coordinates are (x, y) pixel.
    """
    assert len(path) >= 1
    L = len(path)
    frame_positions_cell = np.zeros((num_frames, 2), dtype=np.float32)
    frame_positions_pix = np.zeros((num_frames, 2), dtype=np.float32)
    for f in range(num_frames):
        t = 0.0 if num_frames == 1 else f * (L - 1) / (num_frames - 1)
        k = int(np.floor(t))
        alpha = float(t - k)
        if k >= L - 1:
            ci, cj = path[-1]
            frame_positions_cell[f] = (ci, cj)
        else:
            ci = path[k][0] * (1 - alpha) + path[k + 1][0] * alpha
            cj = path[k][1] * (1 - alpha) + path[k + 1][1] * alpha
            frame_positions_cell[f] = (ci, cj)
        x, y = _cell_center_xy(frame_positions_cell[f, 0], frame_positions_cell[f, 1], cell_px)
        frame_positions_pix[f] = (x, y)
    return frame_positions_cell, frame_positions_pix


def render_frame(
    grid: np.ndarray,
    ball_xy_pix: tuple[float, float],
    goal_ij: tuple[int, int],
    palette: MazePalette,
    cell_px: int,
) -> np.ndarray:
    """Render one RGB frame. Returns uint8 array of shape (H, W, 3)."""
    H_cells, W_cells = grid.shape
    img_h = H_cells * cell_px
    img_w = W_cells * cell_px

    img = Image.new("RGB", (img_w, img_h), palette.passage_rgb)
    draw = ImageDraw.Draw(img)

    # Walls — run-length compress along rows to reduce draw calls.
    for i in range(H_cells):
        j = 0
        while j < W_cells:
            if grid[i, j] == 1:
                j0 = j
                while j < W_cells and grid[i, j] == 1:
                    j += 1
                draw.rectangle(
                    [j0 * cell_px, i * cell_px, j * cell_px - 1, (i + 1) * cell_px - 1],
                    fill=palette.wall_rgb,
                )
            else:
                j += 1

    # Goal marker — filled square a bit smaller than a cell.
    gi, gj = goal_ij
    gcx, gcy = _cell_center_xy(gi, gj, cell_px)
    gr = cell_px / 3.0
    draw.rectangle(
        [gcx - gr, gcy - gr, gcx + gr, gcy + gr],
        fill=palette.goal_rgb,
    )

    # Ball — filled circle.
    bx, by = ball_xy_pix
    br = cell_px / 3.0
    draw.ellipse(
        [bx - br, by - br, bx + br, by + br],
        fill=palette.ball_rgb,
    )

    return np.asarray(img, dtype=np.uint8)


def render_video(
    grid: np.ndarray,
    frame_positions_pix: np.ndarray,
    goal_ij: tuple[int, int],
    palette: MazePalette,
    cell_px: int,
) -> np.ndarray:
    """Render every frame. Returns uint8 array of shape (T, H, W, 3)."""
    frames = [render_frame(grid, (float(x), float(y)), goal_ij, palette, cell_px) for x, y in frame_positions_pix]
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


PROMPT_TEMPLATES: tuple[str, ...] = (
    "A {ball} ball navigates through a {wall} maze from the top-left to the "
    "bottom-right, following the only valid path.",
    "A small {ball} circle moves through a maze with {wall} walls on a "
    "{passage} background, heading toward the {goal} goal marker.",
    "Top-down view of a {passage} maze with {wall} walls; a {ball} ball "
    "traverses the shortest path to the {goal} exit.",
    "A {ball} ball rolls step-by-step through a grid maze, navigating "
    "{wall} corridors to reach the {goal} target at the far corner.",
)


def build_prompt(palette: MazePalette, rng: np.random.Generator) -> str:
    tpl = PROMPT_TEMPLATES[int(rng.integers(0, len(PROMPT_TEMPLATES)))]
    return tpl.format(
        ball=palette.ball_name,
        wall=palette.wall_name,
        passage=palette.passage_name,
        goal=palette.goal_name,
    )


# ---------------------------------------------------------------------------
# End-to-end sample builder
# ---------------------------------------------------------------------------


class MazeSpec(BaseModel):
    """Configuration for one maze sample generator.

    Resolution derived as ``(2*cell_h+1)*cell_px`` by ``(2*cell_w+1)*cell_px``.
    """

    cell_h: int = Field(ge=2)
    cell_w: int = Field(ge=2)
    cell_px: int = Field(ge=4)
    num_frames: int = Field(ge=2)
    palettes: list[MazePalette] = Field(default_factory=lambda: list(DEFAULT_PALETTES))

    @property
    def image_hw(self) -> tuple[int, int]:
        return ((2 * self.cell_h + 1) * self.cell_px, (2 * self.cell_w + 1) * self.cell_px)


class MazeSample(BaseModel):
    """Fully-specified maze sample — everything the reward might need."""

    # Layout
    grid: list[list[int]]  # shape (Hg, Wg), 1=wall 0=passage
    cell_h: int
    cell_w: int
    cell_px: int
    image_h: int
    image_w: int

    # Task
    start: tuple[int, int]  # (i, j) in grid coords
    goal: tuple[int, int]  # (i, j) in grid coords
    path: list[tuple[int, int]]  # BFS shortest path (start → goal inclusive)
    path_len: int

    # Animation
    num_frames: int
    frame_positions_cell: list[tuple[float, float]]  # per-frame (i, j) in cell coords
    frame_positions_pix: list[tuple[float, float]]  # per-frame (x, y) in pixels

    # Palette + text
    palette: MazePalette
    prompt: str

    class Config:
        frozen = False


def build_maze_sample(spec: MazeSpec, rng: np.random.Generator) -> tuple[np.ndarray, MazeSample]:
    """Generate maze + render video + return full metadata.

    Start is always the top-left cell ``(1, 1)``; goal is the bottom-right
    cell ``(2*cell_h - 1, 2*cell_w - 1)``.  A perfect maze guarantees a unique
    path, and BFS finds it in linear time.

    Returns ``(video_uint8, MazeSample)`` where ``video_uint8`` has shape
    ``(num_frames, H, W, 3)``.
    """
    grid = generate_maze_grid(spec.cell_h, spec.cell_w, rng)
    start = (1, 1)
    goal = (2 * spec.cell_h - 1, 2 * spec.cell_w - 1)
    path = bfs_shortest_path(grid, start, goal)

    fp_cell, fp_pix = interpolate_ball_positions(path, spec.num_frames, spec.cell_px)

    palette = spec.palettes[int(rng.integers(0, len(spec.palettes)))]

    video = render_video(grid, fp_pix, goal, palette, spec.cell_px)
    prompt = build_prompt(palette, rng)

    img_h, img_w = video.shape[1], video.shape[2]
    sample = MazeSample(
        grid=grid.astype(int).tolist(),
        cell_h=spec.cell_h,
        cell_w=spec.cell_w,
        cell_px=spec.cell_px,
        image_h=img_h,
        image_w=img_w,
        start=start,
        goal=goal,
        path=[tuple(p) for p in path],
        path_len=len(path),
        num_frames=spec.num_frames,
        frame_positions_cell=[tuple(row) for row in fp_cell.tolist()],
        frame_positions_pix=[tuple(row) for row in fp_pix.tolist()],
        palette=palette,
        prompt=prompt,
    )
    return video, sample
