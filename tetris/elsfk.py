import numpy as np
import random

import copy
import numpy as np

class Brick:
    def __init__(self, body,rotateCount):
        """
        初始化砖块。

        参数:
        body (np.ndarray): 2D 布尔数组，1 表示砖块格子，0 表示空格。
        """
        self.body = np.array(body, dtype=bool)  # 确保是布尔类型
        self.shape = self.body.shape
        self.rotateCount = rotateCount  # 可旋转次数
        self.lt,self.rb = self._get_bounding_box()

    def rotate(self,k):
        """
        旋转砖块 90 度。
        """
        self.body = np.rot90(self.body, k)
        self.shape = self.body.shape
        self.lt,self.rb = self._get_bounding_box()

    def _get_bounding_box(self):
        """
        返回砖块的占用范围（高度, 宽度）。

        返回:
        (int, int),(int, int): 当前砖块实际碰撞箱的左上和右下坐标。
        # 左上包含，右下不包含
        """
        rows = np.any(self.body, axis=1)
        cols = np.any(self.body, axis=0)
        # print(rows,cols)
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1])
        return (top, left), (bottom, right)
    
    def get_bounding_box(self):
        return self.lt,self.rb

    def __str__(self):
        """
        用字符打印砖块形状。
        """
        return '\n'.join(''.join('■' if cell else '.' for cell in row) for row in self.body)
    
    def copy(self):
        """
        返回砖块的深拷贝。
        """
        return Brick(self.body.copy(),self.rotateCount)
        

I = Brick(np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],dtype=bool),2)
O = Brick(np.array([[1, 1], [1, 1]], dtype=bool),1)
T = Brick(np.array([[0, 1, 0], [1, 1, 1],[0,0,0]], dtype=bool),4)
S = Brick(np.array([[0, 1, 1], [1, 1, 0],[0,0,0]], dtype=bool),2)
Z = Brick(np.array([[1, 1, 0], [0, 1, 1],[0,0,0]], dtype=bool),2)
J = Brick(np.array([[0, 0, 1], [1, 1, 1],[0,0,0]], dtype=bool),4)
L = Brick(np.array([[1, 0, 0], [1, 1, 1],[0,0,0]], dtype=bool),4)
TETROMINOS = [I, O, T, S, Z, J, L]
# TETROMINOS = [I,O,I,O,I,O,I]

class TetrisGame:
    """
    Core game logic for Tetris using only numpy operations.
    Grid: 20 rows x 10 columns boolean array.
    """
    def __init__(self, rows=20, cols=10, seed=42):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=bool)
        self.seed = seed
        self.score = 0
        self.level = 1
        self.bag = []
        self.next:Brick
        self.current:Brick
        self.redraw = True
        self.current_pos=[0,0]
        self.rng = random.Random(seed)
        self.spawn_new()

    def bag_7(self):
        # return a shuffled bag of all 7 tetrominos
        tmp = copy.deepcopy(TETROMINOS)
        self.rng.shuffle(tmp)
        return tmp
        
        # tmp = [copy.deepcopy(TETROMINOS)[self.rng.randint(0,6)] for _ in range(7)]
        # return tmp

    def spawn_new(self):
        # ensure bag always has at least 2 pieces so next preview works
        if not self.bag or len(self.bag) <= 1:
            self.bag.extend(self.bag_7())
        # current piece
        self.current = self.bag.pop(0)
        # next piece preview
        self.next = self.bag[0]
        # spawn position
        self.current_pos = [0, (self.cols - self.current.shape[1]) // 2]
        # check game over
        if self._collides(self.current, self.current_pos):
            self.game_over = True
        else:
            self.game_over = False
            
    def _collides(self, shape, pos):
        '''_collides 检测shape是否与board碰撞
        Args:
        shape(Brick): 形状
        pos(list): 位置
        '''
        r, c = pos 
        lp,rp = shape.get_bounding_box()
        if c+lp[1] < 0 or c+rp[1] > self.cols or r+rp[0] > self.rows:
            return True
        board_slice = self.board[r+lp[0]:r + rp[0], c+lp[1]:c + rp[1]]
        return np.any(shape.body[lp[0]:rp[0],lp[1]:rp[1]] & board_slice)


    def rotate(self):
        self.current.rotate(1)
        if self._collides(self.current, self.current_pos):
            self.current.rotate(-1)
            return False
        return True

    def move(self, dx):
        new_pos = [self.current_pos[0], self.current_pos[1] + dx]
        if not self._collides(self.current, new_pos):
            self.current_pos = new_pos
            return True
        return False

    def drop(self):
        while not self._collides(self.current, [self.current_pos[0] + 1, self.current_pos[1]]):
            self.current_pos[0] += 1
        return self.lock()

    def step(self, action):
        '''
            多值返回，drop和Lock返回消除行数，其他返回操作状态
        '''
        # 允许抢断空时间一轮
        if action == 1:
            return self.move(-1)
        elif action == 2:
            return self.move(1)
        elif action == 3:
            return self.rotate()
        elif action == 4:
            return self.drop()
        # gravity
        if not self._collides(self.current, [self.current_pos[0] + 1, self.current_pos[1]]):
            self.current_pos[0] += 1
            return True
        else:
            return self.lock()

    def lock(self):
        r, c = self.current_pos
        lp,rp = self.current.get_bounding_box()
        self.board[r+lp[0]:r + rp[0], c+lp[1]:c + rp[1]] |= self.current.body[lp[0]:rp[0],lp[1]:rp[1]]
        lines = self.clear_lines()
        self.spawn_new()
        self.redraw = True
        return lines

    def clear_lines(self):
        """
        Detect and clear full lines, update score, total lines cleared, level and fall speed.
        Assumes:
        - self.board: np.ndarray of shape (H, W), dtype=bool or {0,1}
        - self.cols: 宽度 W
        - self.score: 当前分数
        - self.lines_cleared: 累计消除行数（初始为 0）
        - self.level: 当前关卡（初始为 1）
        - self.gravity_interval: 下落间隔（秒），随 level 升高而减少
        - self.base_gravity: level=1 时的下落间隔
        """
        H, W = self.board.shape
        # 1. 找出所有要清除的行
        full = np.all(self.board, axis=1)       # 布尔数组，True 表示该行满块
        n_cleared = int(full.sum())
        if n_cleared == 0:
            return 0 # 没有可清除的行

        # 2. 更新分数（按照消除行数表）
        self.score += {1: 100,2: 300,3: 500,4: 800}.get(n_cleared, n_cleared * 200)
        # 每 10 行升一级
        self.level = 1 + (self.score // 1000)
        # 5. 行消除：把满行去掉，顶部补空行，整体下移
        #    ~full 取出未满行的索引
        remaining = self.board[~full]
        # 新板面：顶部补入 n_cleared 行空（False），然后拼上 remaining
        self.board = np.vstack([
            np.zeros((n_cleared, W), dtype=self.board.dtype),
            remaining
        ])
        return n_cleared
        
    def copy(self):
        """
        返回游戏状态的深拷贝。
        """
        new_game = TetrisGame(self.rows, self.cols, self.seed)
        new_game.board = self.board.copy()
        new_game.score = self.score
        new_game.level = self.level
        new_game.bag = [b.copy() for b in self.bag]
        new_game.current_pos = self.current_pos.copy()
        new_game.current = self.current.copy()
        new_game.next = self.next.copy()
        new_game.game_over = getattr(self, 'game_over', False)
        new_game.rng = random.Random(self.seed)  # 重新创建随机数生成器
        return new_game