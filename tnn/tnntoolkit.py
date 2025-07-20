import numpy as np
def place_piece_in_center(piece_body, board_shape=(4, 4)):
    """
    将方块放置到空棋盘的中央位置
    参数:
    piece_body: 方块的body数组 (numpy array)
    board_shape: 棋盘大小，默认(20, 10)
    
    返回:
    centered_board: 包含居中方块的棋盘
    center_pos: 方块在棋盘上的中心位置 (row, col)
    """
    board_height, board_width = board_shape
    piece_height, piece_width = piece_body.shape
    # 计算中心位置
    center_row = (board_height - piece_height) // 2
    center_col = (board_width - piece_width) // 2
    # 创建空棋盘
    centered_board = np.zeros(board_shape, dtype=bool)
    # 将方块放置到中心位置
    centered_board[center_row:center_row + piece_height, center_col:center_col + piece_width] |= piece_body
    return centered_board

def expand_to_4_rotations(X):
    # X: (N, 4, 4) -> (N, 4, 4, 4)
    rotations = [np.rot90(X, k=i, axes=(1, 2)) for i in range(4)]  # 每次绕 (4,4) 平面旋转
    return np.stack(rotations, axis=1)  # (N, 4, 4, 4)
