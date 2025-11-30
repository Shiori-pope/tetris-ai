import numpy as np
import time
from IPython.display import display, HTML, DisplayHandle, clear_output,display_html
# 🟥	🟧	🟨	🟩	🟦	🟪	🟫	⬜
def board_to_html(board):
    # 用 <pre> 保持等宽字体和行结构
    lines = []
    for row in board:
        lines.append(''.join('🟩' if cell else '⬛' for cell in row))
    html = "<pre style='line-height:1.5em;font-size:12px'>" + "\n".join(lines) + "</pre>"
    return HTML(html)

def show_board(board):
    return display_html(board_to_html(board))
# 在 Jupyter 中用 ASCII 实时渲染含当前下落方块的 TetrisGame.board

def render_ascii(handle, game, appendText=None):
    """
    渲染俄罗斯方块游戏状态，包括积分信息
    参数:
    handle: DisplayHandle对象，用于显示内容
    game: TetrisGame实例
    appendText: 可选的附加文本信息
    """
    # 合并已固定方块和当前下落方块
    clear_output(wait=True)
    display = game.board.copy().astype(int)
    shape = game.current
    r0, c0 = game.current_pos
    
    # 添加当前方块到显示矩阵
    for i in range(shape.shape[0]):
        for j in range(shape.shape[1]):
            if shape.body[i, j]:
                rr, cc = r0 + i, c0 + j
                if 0 <= rr < game.rows and 0 <= cc < game.cols:
                    display[rr, cc] = 1
    
    # 生成游戏信息面板
    info_text = f"分数: {game.score} | 等级: {game.level}"
    if appendText:
        info_text += f" | {appendText}"
    
    # 添加下一个方块的预览 (如果需要)
    next_piece_preview = ""
    if hasattr(game, 'next') and game.next is not None:
        next_shape = game.next
        next_piece_preview = "<br>下一个方块:<br>"
        for i in range(next_shape.shape[0]):
            row = []
            for j in range(next_shape.shape[1]):
                if next_shape.body[i, j]:
                    row.append("🟩")
                else:
                    row.append("⬛")
            next_piece_preview += "".join(row) + "<br>"
    
    # 生成HTML
    game_board_html = board_to_html(display).data
    
    # 创建结合游戏板和信息区域的HTML
    full_html = f"""
    <div style="font-family: monospace;">
        <div style="margin-bottom: 10px; font-size: 14px; font-weight: bold;">
            {info_text}
        </div>
        {game_board_html}
        <div style="margin-top: 5px; font-size: 12px;">
            {next_piece_preview}
        </div>
    </div>
    """
    
    handle.display(HTML(full_html))
