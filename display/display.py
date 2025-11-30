import numpy as np
from IPython.display import HTML, DisplayHandle, clear_output, display_html

# Pac-Man glyphs for different cell types
TILE_SYMBOLS = {
    0: "  ",   # empty path
    1: "🟦",  # wall
    2: "• ",  # bean
    3: "  ",  # reserved spawn tiles fall back to empty
    4: "  ",
}
PLAYER_SYMBOL = "😋"
GHOST_SYMBOL = "👻"


def _normalize_state(target):
    """
    将输入统一成字典格式，便于渲染：
    - PacmanGame 实例 -> 调用 state()
    - 包含 map 的 dict -> 直接返回
    - 其他 (如 ndarray) -> 仅渲染地图
    """
    if hasattr(target, "state") and callable(target.state):
        return target.state()
    if isinstance(target, dict) and "map" in target:
        return target

    board = np.array(target)
    return {
        "map": board,
        "player_pos": None,
        "ghost_pos": None,
        "score": 0,
        "total_beans": int(np.count_nonzero(board == 2)),
    }


def board_to_html(map_matrix, player_pos=None, ghost_pos=None):
    board = np.array(map_matrix)
    player = tuple(player_pos) if player_pos is not None else None
    ghost = tuple(ghost_pos) if ghost_pos is not None else None

    lines = []
    for r in range(board.shape[0]):
        row_symbols = []
        for c in range(board.shape[1]):
            if player is not None and (r, c) == player:
                row_symbols.append(PLAYER_SYMBOL)
            elif ghost is not None and (r, c) == ghost:
                row_symbols.append(GHOST_SYMBOL)
            else:
                cell_value = int(board[r, c])
                row_symbols.append(TILE_SYMBOLS.get(cell_value, TILE_SYMBOLS[0]))
        lines.append("".join(row_symbols))

    pre = (
        "<pre style=\"line-height:1.3em;font-size:16px;"
        "font-family:'Segoe UI Emoji',monospace;margin:0\">"
        + "\n".join(lines)
        + "</pre>"
    )
    return HTML(pre)


def show_board(game_or_state):
    """
    直接显示当前局面，可传入：
    - PacmanGame 实例
    - PacmanGame.state() 的字典
    - 仅包含地图的数组
    """
    state = _normalize_state(game_or_state)
    html = board_to_html(state["map"], state.get("player_pos"), state.get("ghost_pos"))
    return display_html(html)


def render_ascii(handle: DisplayHandle, game, append_text=None):
    """
    在 Jupyter 中即时渲染吃豆人局面，同时展示基本信息。
    handle: DisplayHandle，用于更新已有输出
    game: PacmanGame 实例或兼容的 state 字典
    append_text: 附加说明字符串
    """
    clear_output(wait=True)
    state = _normalize_state(game)

    info = f"得分: {state.get('score', 0)} | 剩余豆子: {state.get('total_beans', '?')}"
    if append_text:
        info += f" | {append_text}"

    board_html = board_to_html(
        state["map"],
        state.get("player_pos"),
        state.get("ghost_pos"),
    ).data

    legend = (
        f"图例: {PLAYER_SYMBOL} = Pacman, {GHOST_SYMBOL} = Ghost, "
        "🟦 = Wall, • = Bean"
    )

    full_html = f"""
    <div style="font-family:'Segoe UI', monospace;">
        <div style="margin-bottom:8px;font-size:14px;font-weight:bold;">
            {info}
        </div>
        {board_html}
        <div style="margin-top:6px;font-size:12px;color:#666;">
            {legend}
        </div>
    </div>
    """

    if isinstance(handle, DisplayHandle):
        handle.display(HTML(full_html))
    else:
        display_html(HTML(full_html))
