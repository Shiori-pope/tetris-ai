import logging
from tetris import TetrisGame
from tqdm import tqdm
import numpy as np
import random
import time
import concurrent.futures
import os
kernel = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
# 配置日志记录器
def setup_logger(log_file='yc_final.log', level=logging.INFO):
    logger = logging.getLogger('tetris')
    logger.setLevel(level)
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    
    return logger
logger = setup_logger()

from numpy.lib.stride_tricks import as_strided
def conv2d(input, kernel, padding=0, stride=1):
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
    枚举当前方块所有可能的放置状态（旋转 + 水平位置），并避免重复。
    忽略了soft lock操作，使用布尔掩码

    参数:
      game (TetrisGame): 游戏实例
    返回:
      valid_states: list of ([row, col], rotation)
      footprints: list of board snapshots (np.ndarray)
    """
    # 保存当前状态
    original_piece = game.current.copy()
    original_pos = tuple(game.current_pos)

    H, W = game.rows, game.cols
    valid_states = []
    footprints = []
    seen = set()

    # 遍历每个旋转状态
    for rot in range(game.current.rotateCount):
        # 第一次不旋转，之后每次旋转90度
        if rot > 0:
            game.current.rotate(1)
        # 获取当前方块的有效碰撞箱
        lp, rp = game.current.get_bounding_box()
        # 尝试所有可能的水平位置
        for col in range(-lp[1], game.cols):
            # 从最上方开始
            row = 0
            # 如果起始位置就碰撞，跳过
            if game._collides(game.current, (row, col)):
                continue
            # 下落到最低
            while not game._collides(game.current, (row + 1, col)):
                row += 1

            # 构建落子后的板面指纹
            temp = game.board.copy()
            # 直接用布尔索引填充
            temp[row+lp[0]:row + rp[0], col+lp[1]:col + rp[1]] |= game.current.body[lp[0]:rp[0],lp[1]:rp[1]]
            key = temp.tobytes()
            if key in seen:
                continue
            seen.add(key)
            valid_states.append(([row, col], rot))
            footprints.append(temp)

    # 恢复原始状态
    game.current = original_piece
    game.current_pos = list(original_pos)
    return valid_states, footprints

class GeneticOptimizer:
    def __init__(self, population_size=20, generations=50, mutation_rate=0.2, workers=None):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        # 定义权重范围
        self.weight_ranges = [
            (-1, 3),       # hvar
            (4.0, 50.0),    # layers
            (-0.5, 4.0),    # cty_loss
            (-0.5, 1.0),    # max_height 
            (5.0, 20.0)    # count (负权重)
        ]
        self.best_chromosome = None
        self.best_fitness = float('-inf')
        # 固定评估种子，确保比较公平
        self.evaluation_seeds = [42, 123, 456]
        # 默认使用可用CPU核心数
        self.workers = workers or max(1, os.cpu_count() - 1)
        # 多进程环境下不共享缓存，每个进程维护自己的缓存
        self.fitness_cache = {}
    
    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            chromosome = []
            for min_val, max_val in self.weight_ranges:
                gene = min_val + random.random() * (max_val - min_val)
                chromosome.append(gene)
            population.append(chromosome)
        return population
    
    def _evaluate_single_chromosome(self, chromosome):
        """评估单个染色体的适应度（适合多线程调用）"""
        # 缓存检查 - 注意每个进程有自己的缓存
        chromosome_tuple = tuple(round(g, 4) for g in chromosome)
        if chromosome_tuple in self.fitness_cache:
            return chromosome_tuple, self.fitness_cache[chromosome_tuple]
        
        # 创建自定义损失函数
        def custom_loss(board_bin):
            board = board_bin.astype(np.int8)
            height = np.where((h := board.argmax(axis=0)) != 0, 20 - h, 0)
            cty_loss = np.abs(conv2d(board, kernel, padding=1, stride=1)).sum()/(board.sum() or 1)
            count = board.all(axis=1).sum()
            hvar = height.var()
            remaining = board[~np.all(board, axis=1)]
            # 新板面：顶部补入 n_cleared 行空（False），然后拼上 remaining
            tmpboard = np.vstack([
            np.zeros((count, 10), dtype=board.dtype),
                remaining
            ])
            layers = ((tmpboard[:-1] == 1) & (tmpboard[1:] == 0)).sum()/10
            max_height = height.max()
            real_var = board.sum(axis=0).var()
            special_var = np.sort(height)[1:].var() if np.average(height)-np.min(height)<5 else hvar
            return (special_var * chromosome[0] + 
                   layers * chromosome[1] + 
                   real_var * chromosome[2] + 
                   max_height * chromosome[3] - 
                   count//3 * chromosome[4])
        
        # 对固定种子集运行游戏
        scores = []
        # 换随机种子吧
        for seed in self.evaluation_seeds:
            seeeeed = random.randint(0, 100000)
            game = TetrisGame(seed=seeeeed)
            score = run_game_with_custom_loss(game, custom_loss)
            scores.append(score)
            if score == 0:
                break
        
        # 计算平均分数并缓存
        fitness = sum(scores) / len(scores)
        self.fitness_cache[chromosome_tuple] = fitness
        return chromosome_tuple, fitness
    
    def evaluate_fitness(self, chromosome):
        """评估单个染色体的适应度（游戏得分）- 单线程版本"""
        _, fitness = self._evaluate_single_chromosome(chromosome)
        return fitness
        
    def select_parents(self, population, fitnesses):
        """使用轮盘赌选择法选择父代"""
        total_fitness = sum(max(0, f) for f in fitnesses)
        if total_fitness == 0:
            # 如果所有适应度都是负数，使用排名选择
            sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
            return [population[sorted_indices[0]], population[sorted_indices[1]]]
        
        # 轮盘赌选择
        selection_probs = [max(0, f)/total_fitness for f in fitnesses]
        selected_indices = np.random.choice(len(population), 2, p=selection_probs, replace=False)
        return [population[i] for i in selected_indices]
    
    def crossover(self, parent1, parent2):
        """单点交叉操作"""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def mutate(self, chromosome):
        """变异操作"""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                min_val, max_val = self.weight_ranges[i]
                # 在当前值附近变异，而不是完全随机
                delta = (max_val - min_val) * 0.2  # 变异范围是权重范围的20%
                new_value = chromosome[i] + random.uniform(-delta, delta)
                # 确保在范围内
                chromosome[i] = max(min_val, min(max_val, new_value))
        return chromosome
    
    def optimize(self):
        """执行遗传算法优化过程 - 多线程版本"""
        population = self.initialize_population()
        
        print(f"开始遗传算法优化: 种群大小={self.population_size}, 代数={self.generations}, 工作进程={self.workers}")
        
        for generation in range(self.generations):
            start_time = time.time()
            print(f"第 {generation+1}/{self.generations} 代开始")
            
            # 多线程评估适应度
            fitnesses = []
            chromosomes_to_evaluate = [(i, chrom) for i, chrom in enumerate(population)]

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
                # 提交所有任务
                future_to_idx = {
                    executor.submit(self._evaluate_single_chromosome, chrom): i 
                    for i, chrom in enumerate(population)
                }
                
                # 收集结果，显示进度
                completed = 0
                fitness_dict = {}
                with tqdm(total=len(population), desc=f"Generation {generation+1}") as pbar:
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            _, fitness = future.result()
                            fitness_dict[idx] = fitness
                        except Exception as e:
                            print(f"染色体 {idx} 评估失败: {e}")
                            fitness_dict[idx] = float('-inf')  # 失败的染色体给予最低适应度
                        finally:
                            pbar.update(1)
            
            # 按原始顺序整理适应度结果
            fitnesses = [fitness_dict.get(i, float('-inf')) for i in range(len(population))]
            
            # 找出最佳染色体
            best_idx = fitnesses.index(max(fitnesses))
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_chromosome = population[best_idx].copy()
                print(f"新的最佳权重: {[round(g, 4) for g in self.best_chromosome]}, 得分: {self.best_fitness:.2f}")
                logger.info(f"新的最佳染色体: {self.best_chromosome}, 适应度: {self.best_fitness}")
            
            # 统计信息
            avg_fitness = sum(fitnesses) / len(fitnesses)
            logger.info(f"本代平均适应度: {avg_fitness:.2f}, 最佳适应度: {max(fitnesses):.2f}")
            
            # 创建新一代
            new_population = [self.best_chromosome]  # 精英保留策略
            while len(new_population) < self.population_size:
                # 选择父代
                parents = self.select_parents(population, fitnesses)
                
                # 交叉
                child1, child2 = self.crossover(parents[0], parents[1])
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # 添加到新种群
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
            logger.info(f"第 {generation+1} 代完成，耗时: {time.time() - start_time:.2f}秒")
            
        print(f"优化完成! 最佳权重: {[round(g, 4) for g in self.best_chromosome]}")
        return self.best_chromosome, self.best_fitness

def run_game_with_custom_loss(game, loss_function):
    """运行一局游戏，使用自定义损失函数，自动提前终止无希望的游戏"""
    max_steps = 5000  # 限制最大步数
    step = 0
    last_score = 0
    no_progress_count = 0
    lines_bonus = 0
    while not getattr(game, 'game_over', False) and step < max_steps:
        # 枚举当前方块所有可能的放置状态
        valid_states, readable_footprints = enumerate_GameStatus(game)
        if not valid_states:
            break  # 没有可行的放置状态
        # 计算每种状态的loss
        best_loss = float('inf')
        best_idx = -1
        for i,_ in enumerate(valid_states):
            current_loss = loss_function(readable_footprints[i])
            if current_loss < best_loss:
                best_loss = current_loss
                best_idx = i
        
        if best_idx == -1:
            break  # 没有找到好的状态
            
        # 获取并执行最佳移动
        (best_row, best_col), best_rot = valid_states[best_idx]
        for _ in range(best_rot):
            if not game.step(3): break  # 旋转
        while game.current_pos[1] < best_col:
            if not game.step(2): break  # 右移
        while game.current_pos[1] > best_col:
            if not game.step(1): break  # 左移
        lines = game.step(4)  # 快速下落
        # 连消额外奖励 2行600 3行1000 4行5000
        if lines == 2:
            lines_bonus += 600
        elif lines == 3:
            lines_bonus += 1000
        elif lines == 4:
            lines_bonus += 5000
        # 检测长时间无进展情况（提前终止表现不佳的染色体）
        if game.score == last_score:
            no_progress_count += 1
            if no_progress_count > 25:  # 25步无得分增长
                break
        else:
            no_progress_count = 0
            last_score = game.score
        
        step += 1
        
    return game.score  # 返回总得分

if __name__ == "__main__":

    optimizer = GeneticOptimizer(population_size=100, generations=100, mutation_rate=0.1)
    best_weights, best_score = optimizer.optimize()
    print("优化完成!")
    print(f"最佳权重: {best_weights}")
    print(f"最佳得分: {best_score}")