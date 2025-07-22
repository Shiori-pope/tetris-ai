# 俄罗斯方块强化学习项目

## 项目简介

这是一个基于深度学习的俄罗斯方块AI项目，集成了启发式算法、神经网络和强化学习技术。项目实现了从传统启发式搜索到深度Q学习(DQN)的完整技术栈，最终能够训练出可以永续游玩俄罗斯方块的AI。

## Demo

![Demo Animation](assets/demo.gif)

## 项目架构

```
a-my-elsfk/
├── tetris/              # 俄罗斯方块游戏核心
│   ├── elsfk.py        # 纯NumPy实现的俄罗斯方块引擎
│   └── toolF.py        # 游戏工具函数
├── tnn/                # 神经网络模型
│   ├── tnn.py          # TetrisNN - CNN特征提取网络
│   ├── tnn_m.py        # TNN-M - 改进版模型  
│   ├── tnn_ms.py       # TNN-MS - 多尺度特征版本
│   ├── tnnextreme.py   # TNN-Extreme - 极致性能版本
│   └── tnntoolkit.py   # 神经网络工具函数
├── dataset_generator/   # 训练数据生成
│   ├── da.py           # 启发式算法数据生成器
│   └── datapaster.py   # 数据处理和转换工具
├── display/            # 可视化模块
│   └── display.py      # Jupyter显示接口
├── gs.py               # 遗传算法启发式搜索
├── gs_withslide.py     # 带滑行功能的启发式搜索
└── *.ipynb             # Jupyter训练和实验笔记本
```

## 核心技术

### 1. 游戏引擎 (`tetris/elsfk.py`)

- **纯NumPy实现**：100μs级别的超高性能帧时间
- **完整游戏逻辑**：支持7种标准俄罗斯方块，包括旋转、碰撞检测、消行等
- **7-bag随机化**：符合现代俄罗斯方块标准的方块生成机制
- **种子控制**：支持确定性重放和测试

### 2. 启发式搜索算法 (`gs.py`, `gs_withslide.py`)

#### 核心算法特性：
- **枚举式搜索**：为每个方块生成所有可能的放置位置（约30个状态）
- **两步贪心**：考虑当前方块和下一个方块的联合优化
- **滑行技术**：实现方块的侧向滑行操作，增强放置灵活性
- **遗传算法优化**：自动优化启发式函数参数

#### 损失函数设计：
```python
# 高度惩罚 + 洞穴惩罚 + 不平整度 + 多行消除奖励
loss = height_penalty + hole_count + bumpiness - line_clear_bonus
```

#### 性能表现：
- **永续游戏**：两步贪心算法实现理论上的无限游戏
- **高效计算**：平均每步计算时间 < 100μs
- **高得分**：单局游戏可达10000+分

### 3. 神经网络架构

#### TetrisNN - 基础CNN模型 (`tnn/tnn.py`)
```python
输入：
- 游戏板面 (20×10)
- 当前方块 (4×4) 
- 下一个方块 (4×4)

特征提取：
- FrontEndNet: Inception块 + 卷积层处理板面
- miniNN: 小型CNN处理方块信息

多头输出：
- translation_head: 水平移动 (-4到5，10分类)
- rotation_head: 旋转状态 (0-3，4分类)  
- slide_head: 滑行方向 (-1,0,1，3分类)
```

#### TNN变体：
- **TNN-M**: 改进的特征融合架构
- **TNN-MS**: 多尺度特征版本，增强细粒度特征捕获
- **TNN-Extreme**: 极致性能版本，优化推理速度

### 4. 强化学习 (DQN)

#### TetrisDQN (`DQN.ipynb`)
- **Q网络**：基于TetrisNN的Q值估计网络
- **经验回放**：稳定训练过程
- **目标网络**：减少训练不稳定性
- **预训练迁移**：从监督学习模型初始化

#### 训练环境：
- **状态空间**：游戏板面 + 当前/下一个方块
- **动作空间**：120个离散动作 (10×4×3组合)
- **奖励设计**：基于消行数、生存时间和游戏得分

### 5. 数据生成管道

#### 启发式数据生成 (`dataset_generator/da.py`)
```python
# 多进程并行生成
seeds = random.sample(range(100000), 110)  
for seed in seeds:
    game = TetrisGame(seed=seed)
    operations = heuristic_play(game, max_steps=10000)
    save_to_file(f"games_{seed}.txt", operations)
```

#### 数据规模：
- **120万训练样本**：110个种子 × 每个约10000步
- **数据格式**：`[board_state, current_piece, next_piece] -> [move, rotation, slide]`
- **预处理**：99%数据保留，去除游戏结束状态

## 实验结果

### 模型性能对比

| 模型 | 参数量 | 训练数据 | 平均得分 | 说明 |
|-----|--------|----------|----------|------|
| 启发式(两步贪心) | - | - | 无限/永续 | 理论最优 |
| TetrisNN | 1.2M | 120K样本 | 8000+ | 基础CNN模型 |  
| TNN-3 | 1.2M | 120K样本 | 8436+ | 改进架构 |
| TNN-Extreme | 1.2M | 120K样本 | 8436+ | 极致优化版 |
| DQN | 变长 | 在线学习 | 持续提升 | 强化学习版本 |

### 关键指标
- **帧时间**：100μs (游戏逻辑) + 32.7μs (卷积计算)
- **训练速度**：GPU加速下约6.51步/秒
- **收敛性**：监督学习模型可在50epoch内收敛

## 技术亮点

### 1. 高性能游戏引擎
- 纯NumPy实现，无第三方游戏库依赖
- 微秒级帧时间，支持高频训练
- 完整支持现代俄罗斯方块特性

### 2. 多层次AI架构
- **Level 1**: 启发式搜索 (贪心算法)
- **Level 2**: 监督学习 (模仿启发式)  
- **Level 3**: 强化学习 (自主优化)

### 3. 创新的滑行机制
实现了方块的侧向滑行，增加了游戏策略的复杂度和灵活性。

### 4. 端到端可视化
集成Jupyter显示系统，支持训练过程和游戏过程的实时可视化。

## 运行方式

### 环境要求
```bash
Python 3.7+
torch >= 1.8.0
numpy >= 1.19.0  
tqdm
matplotlib (可视化)
jupyter (交互式训练)
```

### 快速开始

1. **运行启发式AI**：
```python
from tetris import TetrisGame
from gs import heuristic_play

game = TetrisGame(seed=42)
operations = heuristic_play(game, max_steps=10000)
print(f"游戏结束，得分: {game.score}")
```

2. **训练神经网络**：
```python
# 在 tnn.ipynb 中运行
from tnn import TetrisNN
model = TetrisNN(dropout_rate=0.3)
# ... 训练代码
```

3. **强化学习训练**：
```python
# 在 DQN.ipynb 中运行
trainer = DQNTrainer(dqn_model, target_model, env, config)
trainer.train()
```

4. **可视化游戏**：
```python
# 在 tnndisplay.ipynb 中运行
predictor = TetrisPredictor('best_tetris_model.pth')
# ... 可视化代码
```

## 文件说明

### 模型文件
- `tetris_model1.2M.pth` - TNN基础模型
- `tetris_model3_1.2M.pth` - TNN-3模型  
- `tetris_modelE_1.2M.pth` - TNN-extreme模型
- `84.36+games.pth` - 120K数据集训练模型
- `6dqn_tetris*.pth` - DQN强化学习模型

### 实验记录
- `devlog.md` - 开发日志和实验记录
- `*.log` - 训练日志文件
- `pictures/` - 实验结果图表

## 未来方向

1. **Transformer架构**：引入注意力机制处理序列信息
2. **多任务学习**：同时优化游戏性能和视觉美观度
3. **强化学习优化**：探索更高效的奖励函数设计
4. **实时对战**：开发多智能体对战系统

## 贡献者

本项目由深度学习和强化学习算法实现，展示了从传统搜索算法到现代神经网络的完整技术演进路径。

---

*该项目体现了AI在经典游戏中的应用，从启发式算法的确定性方案到强化学习的自适应学习，展示了不同AI技术的互补性和渐进式发展过程。*
