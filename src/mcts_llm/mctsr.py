from __future__ import annotations

"""

Implements the MCTS + Self-Refine algorithm from
`Accessing GPT-4 level Mathematical Olympiad Solutions via Monte
Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report`
by Zhang et. al.

The authors' [repo](https://github.com/trotsky1997/MathBlackBox) uses critiques,
refinements, and parent nodes' answers as conversation history.
I haven't tried it yet.

MCTS 算法是一种基于蒙特卡洛树搜索的强化学习算法，用于解决决策过程中的探索和利用问题,包含以下4个步骤:
• Selection: Starting from the root, the algorithm navigates through promising child nodes
based on specific strategies (e.g., UCT), continuing until a leaf node is reached.
• Expansion: At the leaf node, unless it represents a terminal state of the game, one or more
feasible new child nodes are added to illustrate potential future moves.
• Simulation or Evaluation: From the newly added node, the algorithm conducts random simulations—often termed "rollouts"—by selecting moves arbitrarily until a game’s conclusion
is reached, thereby evaluating the node’s potential.
• Backpropagation: Post-simulation, the outcome (win, loss, or draw) is propagated back to
the root, updating the statistical data (e.g., wins, losses) of each traversed node to inform
future decisions.
"""

import random
import math
from collections import deque
from enum import Enum
from pydantic import BaseModel
import tqdm
import numpy as np

ROOT_UCT_SCORE = 10_000

"""
BaseModel 类提供了自动数据验证的功能。当你创建一个基于 BaseModel 的子类并且为其字段添加类型注解（type annotations），Pydantic 会在实例化对象时自动校验给定的数据是否符合预期类型和约束。
"""
class MCTSNode(BaseModel):
    """
    MCTSNode 类定义了MCTS中的节点结构，包含答案、父节点、子节点列表、访问次数、Q值和奖励样本列表。
    通过add_child 方法可以添加子节点，通过add_reward 方法可以添加奖励样本并更新Q值。__repr__ 方法提供了节点的字符串表示形式，
    便于调试和日志记录。这些属性和方法共同支持MCTS算法中的节点管理和状态更新。
    """
    answer: str  # 存储节点的答案。 eg: "我不知道。"
    parent: MCTSNode | None = None # 存储节点的父节点。如果节点是根节点，则父节点为 None。
    children: list[MCTSNode] = [] #  存储节点的子节点列表。
    visits: int = 0 # 存储节点的访问次数。
    Q: float = 0 # 存储节点的Q值，表示节点的质量。
    reward_samples: list[int] = [] # 存储节点的奖励reward样本列表, 是具体的reward值。

    def add_child(self, child_node: MCTSNode):
        """向当前节点添加一个子节点。"""
        self.children.append(child_node)

    def __repr__(self):
        """返回节点的字符串表示形式，用于调试和日志记录。通过eval(repr(node)) 可以将字符串表示形式转换为MCTSNode对象。"""
        return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits})"

    def add_reward(self, reward: int):
        """向节点添加一个奖励reward，并更新节点的Q值。"""
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples) # 平均值
        min_reward = np.min(self.reward_samples) # 最小值

        # Average worst-case and average outcomes
        self.Q = (min_reward + avg_reward) / 2


class SelectionPolicy(Enum):
    GREEDY = 1  # 贪婪选择。
    IMPORTANCE_SAMPLING = 2  # 重要性采样。
    PAIRWISE_IMPORTANCE_SAMPLING = 3  # 成对重要性采样。

class InitializeStrategy(Enum):
    ZERO_SHOT = 1  # 生成一个零样本答案。
    DUMMY_ANSWER = 2  # 使用虚拟答案（如“我不知道”）。

"""
这些参数共同定义了 MCTSr 算法的配置和状态信息，包括问题的描述、算法的运行参数、节点的选择和初始化策略、以及日志记录等。
通过调整这些参数，可以控制算法的探索和利用行为，优化搜索过程，并记录算法的运行状态以供分析和调试。
"""
class MCTSr(BaseModel):
    problem: str  # 存储当前要解决的数学问题。
    max_rollouts: int  # 定义算法运行的最大迭代次数（rollouts）
    exploration_constant: float = 1.0  # 控制UCT公式中探索项的权重。较大的值鼓励更多的探索(Exploration)，较小的值鼓励更多的利用(Exploitation)。
    max_children: int = 2  # 定义每个节点最多可以扩展的子节点数量。
    epsilon: float = 1e-10  # 在计算UCT值时，用于避免除零错误的小常数。
    reward_limit: int = 95  # 定义奖励的上限。超过这个值的奖励会被减少。
    excess_reward_penalty: int = 5  # 超过奖励上限时，减少的惩罚值。
    selection_policy: SelectionPolicy = SelectionPolicy.IMPORTANCE_SAMPLING  # 定义节点选择策略。可以是贪婪选择、重要性采样或成对重要性采样。
    initialize_strategy: InitializeStrategy = InitializeStrategy.ZERO_SHOT  # 定义根节点的初始化策略。可以是零样本生成或使用虚拟答案。

    root: MCTSNode = MCTSNode(answer="I don't know.")  # 定义搜索树的根节点。默认情况下，根节点的答案是“我不知道”。在实际情况中，也可以是prompt本身

    # Logs
    # 存储所有生成的批评（critiques）。
    # critiques = ["答案缺少详细推理步骤", "计算错误"]
    critiques: list[str] = []

    # 存储所有生成的优化（refinements）。
    # 如refinements = ["修正后的答案", "添加了详细推理步骤"]
    refinements: list[str] = []

    rewards: list[float] = []  # 存储所有生成的奖励（rewards）。
    selected_nodes: list[MCTSNode] = []  # 存储所有被选中的节点。

    """
    下面的两个方法是由子类来实现的，后面会给出相应的子类实现方法，它们用于定义自我优化（Self-Refine）和自我评估（Self-Evaluation）的具体实现。
    由于这些方法的具体实现依赖于不同的模型（如LLaMA-3 8B或GPT-4），因此在基类中它们被定义为抽象方法，需要在具体的子类中实现。
    """
    def self_refine(self, node: MCTSNode) -> MCTSNode:
        raise NotImplementedError()

    def _evaluate_answer(self, node: MCTSNode) -> int:
        raise NotImplementedError()

    def self_evaluate(self, node: MCTSNode):
        """Evaluate the quality of the answer. Sample `num_samples` times and average the results."""
        reward = self._evaluate_answer(node)

        if reward > self.reward_limit:
            reward -= self.excess_reward_penalty
        # 先refine->evaluate->更新reward
        node.add_reward(reward)

    """
    反向传播的目的是在模拟结束后，将模拟结果（如奖励值）从叶节点反向传播到根节点，更新每个节点的统计信息（如Q值和访问次数）。
    """
    def backpropagate(self, node: MCTSNode):
        parent = node.parent  # 获取当前节点的父节点。
        # 从当前节点开始，向上遍历直到根节点(Root)，更新每父节点的统计信息。
        while parent:
            # 找到父节点的所有子节点中Q值最高的那个。
            best_child_Q = max(child.Q for child in parent.children)
            # 更新父节点的Q值，通过平均当前Q值和最佳子节点的Q值来计算。
            parent.Q = (parent.Q + best_child_Q) / 2
            # 增加父节点的访问次数。(反向传播backpropagate算一次访问)
            parent.visits += 1
            # 将父节点更新为其父节点，继续向上遍历。
            parent = parent.parent

    """
    Upper Confidence Bound
    """
    def uct(self, node: MCTSNode):
        if not node.parent: # 根结点
            # Using an arbitrarily high UCT score for the root node.
            # helps to prioritize breadth. 广度优先
            return ROOT_UCT_SCORE

        return node.Q + self.exploration_constant * math.sqrt(math.log(node.parent.visits + 1) / (node.visits + self.epsilon))

    """
    下面的代码是用于判断一个节点是否已经完全扩展。完全扩展的节点意味着它已经达到了最大子节点数量，或者它的子节点中至少有一个节点的Q值超过了当前节点的Q值。
    """
    def is_fully_expanded(self, node: MCTSNode):
        return len(node.children) >= self.max_children or any(child.Q > node.Q for child in node.children)

    """
    这段代码实现了MCTS中的节点选择过程，通过遍历搜索树中的节点，找到未完全扩展且具有最高UCT值的节点。

    初始化候选节点列表和待考虑节点队列。
    遍历待考虑节点队列，检查每个节点是否已经完全扩展，并将未完全扩展的节点添加到候选节点列表中。
    如果候选节点列表为空，则返回根节点。
    根据选择策略从候选节点列表中选择一个节点，包括贪婪选择、重要性采样和成对重要性采样。 通过这些步骤，算法可以在搜索过程中选择最有潜力的节点进行进一步的探索和优化。
    """
    def select_node(self)->MCTSNode:
        """Select a non-fully expanded node with the highest UCT value.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own
        """
        # 初始化一个候选节点列表 candidates 和一个待考虑节点队列 to_consider。并将根节点 self.root 添加到 to_consider 队列中
        candidates: list[MCTSNode] = []
        to_consider = deque([self.root]) # 双端队列

        # 遍历待考虑节点队列，检查每个节点是否已经完全扩展，并将未完全扩展的节点添加到候选节点列表中。
        while to_consider:
            current_node = to_consider.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)

        # 如果候选节点列表为空，则返回根节点。
        if not candidates:
            return self.root

        # 根据选择策略从候选节点列表中选择一个节点。
        if self.selection_policy == SelectionPolicy.GREEDY:
            return max(candidates, key=self.uct)
        elif self.selection_policy == SelectionPolicy.IMPORTANCE_SAMPLING:
            # 重要性采样，根据UCT值对候选节点进行加权采样。计算每个候选节点的UCT值，并使用 random.choices 函数进行加权采样。
            # Sample, weighted by UCT score
            uct_scores = [self.uct(node) for node in candidates]
            selected_pair_idx = random.choices(range(len(candidates)), weights=uct_scores, k=1)[0]
            return candidates[selected_pair_idx]
        elif self.selection_policy == SelectionPolicy.PAIRWISE_IMPORTANCE_SAMPLING:
            # 成对重要性采样：
            #   根据UCT值的差异对候选节点进行成对加权采样。
            #   计算每个候选节点的UCT值，并生成所有节点对的列表。
            #   计算每对节点的UCT值差异，并使用 random.choices 函数进行加权采样。
            #   选择UCT值较高的节点。
            # Sample, weighted by the difference in UCT scores between pairs
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [
                (i, j) for i in range(len(candidates)) for j in range(len(candidates))
            ]
            pair_weights = [
                max(uct_scores[i], uct_scores[j]) - min(uct_scores[i], uct_scores[j])
                for i, j in pairs
            ]
            selected_pair_idx = random.choices(range(len(pairs)), weights=pair_weights, k=1)[0]
            selected_candidate_idx = max(
                pairs[selected_pair_idx], key=lambda x: uct_scores[x]
            )
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    def zero_shot(self) -> str:
        """Generate a zero-shot answer."""
        raise NotImplementedError()

    def initialize(self):
        """Generate a zero-shot answer."""
        if self.initialize_strategy == InitializeStrategy.ZERO_SHOT:
            self.root = MCTSNode(answer=self.zero_shot())
        elif self.initialize_strategy == InitializeStrategy.DUMMY_ANSWER:
            self.root = MCTSNode(answer="I don't know.")
        else:
            raise ValueError(f"Invalid initialize strategy: {self.initialize_strategy}")

    def run(self):
        self.initialize()
        for _ in tqdm.tqdm(range(self.max_rollouts)):
            node = self.select_node()
            self.self_evaluate(node)
            child = self.self_refine(node)
            node.add_child(child)
            self.self_evaluate(child)
            self.backpropagate(child)

        return self.get_best_answer()

    def get_best_answer(self):
        from collections import deque

        to_visit = deque([self.root])
        best_node = self.root

        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)

        return best_node.answer

    def print(self):
        print_tree(self.root)


def print_tree(node: MCTSNode | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for child in node.children:
        print_tree(child, level + 1)

