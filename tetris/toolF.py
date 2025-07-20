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
def conv2d(input, kernel, padding=0, stride=1):
    from numpy.lib.stride_tricks import as_strided
    H, W = input.shape
    h, w = kernel.shape
    p, s = padding, stride

    # 填充
    if p > 0:
        input = np.pad(input, ((p, p), (p, p)), mode='constant')
    H_p, W_p = input.shape

    # 计算输出尺寸
    out_h = (H_p - h) // s + 1
    out_w = (W_p - w) // s + 1

    # 构造窗口视图
    shape = (out_h, out_w, h, w)
    strides = (s*input.strides[0], s*input.strides[1],
               input.strides[0],   input.strides[1])
    patches = as_strided(input, shape=shape, strides=strides)

    # (out_h, out_w, h, w) 与 (h, w) 做广播乘法再求和
    return np.einsum('ijmn,mn->ij', patches, np.flipud(np.fliplr(kernel)))

def enumerate_GameStatus(game):
    """
    枚举当前方块所有可能的有效放置状态（旋转+位置）
    
    参数:
    game (TetrisGame): 游戏实例
    
    返回:
    ### list: 
    包含元组([row, col], rotation, slide)的列表，每个元素表示一种可能的放置状态 slide -1 0 1 一阶softlock的位置
    ### list:
    包含所有可能的板面
    """
    from .elsfk import TetrisGame
    from copy import deepcopy
    # 保存当前方块状态，以便函数结束时恢复
    original_body = deepcopy(game.current)
    original_pos = list(game.current_pos)
    # 使用放置后的游戏板状态作为唯一标识进行去重
    # footprint -> pos rot slide board
    footprint_map = {}
    # 尝试所有可能的旋转状态
    for rot in range(game.current.rotateCount):
        # 第一次不旋转，之后每次旋转90度
        if rot > 0:
            game.current.rotate(1)
        # 获取当前方块的有效碰撞箱
        lp, rp = game.current.get_bounding_box()
        # 尝试所有可能的水平位置
        for col in range(-lp[1], game.cols):
            # 设置初始位置（最顶部）
            row = 0
            pos = [row, col]
            # 如果初始位置就发生碰撞，跳过此位置
            if game._collides(game.current, pos):
                continue
            # 使用广度优先搜索探索所有可能的落点
            positions_to_check = [(pos, 0, 0)]  # (位置, 已尝试滑行次数, 滑行方向)
            while positions_to_check:
                current_pos, slide_count, slide_direct = positions_to_check.pop(0)
                current_pos[1] += slide_direct
                # 下落到最低点
                while not game._collides(game.current, [current_pos[0] + 1, current_pos[1]]):
                    current_pos[0] += 1
                # 记录当前位置的方块放置情况
                temp_board = game.board.copy()
                r, c = current_pos
                # board切出碰撞箱所在区域做或运算
                temp_board[r+lp[0]:r + rp[0], c+lp[1]:c + rp[1]] |= game.current.body[lp[0]:rp[0],lp[1]:rp[1]]
                
                # 去重检查
                board_footprint = temp_board.tobytes()
                if board_footprint not in footprint_map or (footprint_map[board_footprint][2] != 0 and slide_direct == 0):
                    # 记录当前状态
                    footprint_map[board_footprint] = [[current_pos[0],current_pos[1]-slide_direct], rot, slide_direct,temp_board]
                
                # 如果滑行次数未超限，尝试左右滑动
                if slide_count < 1:  # 限制滑行次数，防止无限循环
                    # 尝试左移
                    left_pos = [current_pos[0], current_pos[1] - 1]
                    if not game._collides(game.current, left_pos):
                        # 均传入当前坐标 把slide当成offset计算实际坐标
                        positions_to_check.append((current_pos.copy(), slide_count + 1, -1))
                    # 尝试右移
                    right_pos = [current_pos[0], current_pos[1] + 1]
                    if not game._collides(game.current, right_pos):
                        positions_to_check.append((current_pos.copy(), slide_count + 1, 1))
    # 恢复方块到原始状态
    game.current = original_body
    game.current_pos = original_pos
    valid_states = []
    readable_footprints = []
    for v in footprint_map.values():
        valid_states.append([v[0], v[1], v[2]])
        readable_footprints.append(v[3])
    return valid_states,readable_footprints