# mcts-llm

## MCTSr

Based on [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394) by Zhang, et al.

At a high level, MCTSr iteratively generates solutions to a specified (math) problem.

In a MCTSr tree, nodes correspond to attempted answers, and edges correspond to attempts to improve the answer.


### Initialize
Generate an solution to the problem. This paper uses a "dummy" solution (e.g. `"I don't know"`).

### Select a node to expand
We gather a set of candidate nodes which haven't been fully expanded.

A node is fully expanded if either:
1. it has `max_children`
2. any of its children have a Q value which is greater than its own

Once we've gathered the candidates, we compute UCT scores for each candidate node.
There are a few ways we can make our selection:
1. Greedily (choose the node with the highest UCT)
2. Importance sampling (sample from the set of candidates, weighted by their UCT score)
3. Pairwise importance sampling (sample the max from a pair of nodes from the set of candidates, weighted by the difference between the pair's UCT scores)

The authors mention that they perform greedy selection in the paper. In their [repo](https://github.com/trotsky1997/MathBlackBox/blob/main/gen_mcts_dpo.py#L182), they also perform pairwise sampling and save the (question, answer1, answer2) tuples for use in DPO.

### Expand the node

Expansion involves several steps:
1. Generate a critique of the current solution.
2. Refine the solution based on the critique.
3. Add a new child, corresponding to the refined solution.
4. Self-evaluate the `reward` of the new child.
5. Backpropagate the reward from the new child through its parents, through to the root.


# Results
I haven't run extensive evals on this.


## [AIME 2024](./results/AIME_2024_llama_3_8b.csv)
- `max_rollouts=8`
- `max_children=2`


# 推理问题

1. 问题1
User: S先生、P先生、Q先生他们知道桌子的抽屉里有16张扑克牌：红桃A、Q、4 黑桃J、8、4、2、7、3 草花K、Q、5、4、6 方块A、5。约翰教授从这16张牌中挑出一张牌来，并把这张牌的点数告诉 P先生，把这张牌的花色告诉Q先生。这时，约翰教授问P先生和Q 先生：你们能从已知的点数或花色中推知这张牌是什么牌吗？于是，S先生听到如下的对话：

P先生：我不知道这张牌。

Q先生：我知道你不知道这张牌。

P先生：现在我知道这张牌了。

Q先生：我也知道了。

请问：这张牌是什么牌？

QwQ-32B-Preview
这道题是逻辑推理题，涉及到信息的逐步披露和推理。题目中有一副16张扑克牌，分别是：

红桃：A、Q、4

黑桃：J、8、4、2、7、3

草花：K、Q、5、4、6

方块：A、5

约翰教授选了一张牌，把点数告诉了P先生，把花色告诉了Q先生。然后通过他们的对话，我们要推断出是哪张牌。

首先，让我们来梳理一下对话的内容：

P先生说：“我不知道这张牌。”

这意味着点数是重复的，因为如果点数唯一，P先生就能确定是哪张牌。例如，如果点数是K，只有草花K，那P先生就能确定是草花K。但P先生说不知道，所以点数必须在多个花色中出现。看看哪些点数是重复的：

A：红桃和方块

Q：红桃和草花

4：红桃、黑桃和草花

5：草花和方块

其他点数都是唯一的在某个花色中。

所以，可能的点数是A、Q、4、5。

Q先生说：“我知道你不知道这张牌。”

这句话有点 tricky。Q先生知道花色，而且他说他知道P先生不知道。这意味着，根据Q先生知道的花色，可以确定点数一定是重复的。因为如果花色中的点数是唯一的，那么P先生就能知道是哪张牌。但Q先生肯定地说他知道P先生不知道，所以花色中的点数必须是重复的。

让我们看看每个花色中的点数情况：

红桃：A、Q、4

A：红桃和方块

Q：红桃和草花

4：红桃、黑桃、草花

所以，红桃花色中的所有点数都是重复的。

黑桃：J、8、4、2、7、3

4：黑桃、红桃、草花

其他点数在黑桃中是唯一的。

但Q先生说他知道P先生不知道，所以花色中的所有点数都必须是重复的。但在黑桃中，除了4，其他点数都是唯一的。所以，如果花色是黑桃，且点数是J、8、2、7、3，那么P先生就能知道，这与Q先生的说法矛盾。因此，花色不能是黑桃。

草花：K、Q、5、4、6

Q：草花和红桃

4：草花、红桃、黑桃

5：草花和方块

K和6是唯一的。

所以，如果花色是草花，点数是K或6，P先生就能知道，这又与Q先生的说法矛盾。因此，花色不能是草花。

方块：A、5

A：方块和红桃

5：方块和草花

所以，方块中的点数都是重复的。

综上所述，Q先生知道花色，且确定所有可能的点数都是重复的，所以花色只能是红桃或方块。

P先生说：“现在我知道这张牌了。”

P先生原来不知道，但现在知道了。这意味着，在Q先生发言后，P先生通过点数和花色的可能组合，确定了是哪张牌。

我们已经知道花色只能是红桃或方块，点数是A、Q、4、5。

但P先生现在知道牌了，说明点数加上花色的限制，使得只有唯一一个可能性。

让我们看看点数在红桃和方块中的分布：

A：红桃和方块

Q：只有红桃

4：红桃

5：方块

等一下，Q先生说花色只能是红桃或方块，但根据之前的分析，花色是红桃或方块。

但是，点数Q只有红桃，4只有红桃，A有红桃和方块，5只有方块。

P先生原来不知道，说明点数是重复的，即A或5。

如果点数是Q或4，只有红桃，那么P先生知道点数后，结合花色是红桃，就能知道是哪张牌，这与P先生第一句“我不知道这张牌”矛盾。

所以，点数不能是Q或4，只能是A或5。

现在，P先生说他知道牌了，说明在A或5中，结合花色，只有唯一可能性。

但A在红桃和方块都有，5只有方块有。

等一下，如果点数是5，只有方块5，那么P先生知道点数是5，结合花色是方块，就可以确定是方块5。

如果点数是A，花色可以是红桃或方块，那么P先生还是不能确定是哪一张。

但是，P先生说现在知道这张牌了，所以点数不能是A，因为A在红桃和方块都有，除非花色是方块，但P先生只有点数信息。

等一下，也许我需要更仔细地思考。

Q先生说花色是红桃或方块，因为只有这些花色的点数都是重复的。

P先生知道点数，现在他说知道了，说明在花色是红桃或方块的限制下，点数唯一确定了牌。

对于点数A，花色可以是红桃或方块，所以P先生不能确定是哪一张。

对于点数5，只有方块5，所以P先生知道是方块5。

因此，只有当点数是5时，P先生才能确定是方块5。

所以，牌是方块5。

Q先生说：“我也知道了。”

Q先生原来只知道花色，但不知道是哪张牌，现在他也知道了。

如果花色是方块，点数是A或5。

但P先生刚才已经确定是方块5，所以Q先生现在也知道了是方块5。

如果花色是红桃，点数是A、Q、4。

但是，根据之前的推理，点数Q和4只有红桃，点数A有红桃和方块。

但P先生原来不知道，所以点数不能是Q或4，只能是A。

但是，如果花色是红桃，点数是A，P先生不能确定是红桃A还是方块A。

但P先生说现在知道了，所以只有当点数是5，花色是方块时，P先生才能确定。

因此，Q先生现在也确定是方块5。

综上所述，这张牌是方块5。


2. 问題2
User
有个人想买几套餐具，到了店里发现，自己的钱可以买21把叉子和21个勺子，也够买28个小刀，但是，他想三样东西都买，而且要配成一套，并且把钱刚好花完，如果你是这个人，你会怎么买呢？
QwQ-32B-Preview
这个问题挺有意思的。我得好好想想。题目说，我有多少钱可以买21把叉子和21个勺子，或者可以买28个小刀。但是我想三样都买，而且要配成一套，还要把钱正好花完。我要怎么安排呢？

首先，我需要理解一下这里的“一套”是什么意思。

我想应该是买一把叉子、一个勺子和一个小刀作为一个组合，也就是一套包括这三种餐具各一件。对吧？

所以，我的目标是买若干套这样的餐具，每套包含一把叉子、一个勺子和一个小刀，而且要把所有的钱都花完。

但是，题目中给出的信息是，我的钱可以买21把叉子和21个勺子，或者可以买28个小刀。

这说明，叉子和勺子的价格是相同的，因为用同样的钱可以买到相同数量的叉子和勺子，而小刀的价格不同，因为同样的钱可以买更多的小刀。

让我设一下变量，方便计算。

假设叉子的价格是x，勺子的价格也是x，小刀的价格是y。

那么，根据题意：21x + 21x = 42x，这是我的总预算。

另外，这个总预算也可以买28个小刀，也就是28y。

所以，42x = 28y。从这个等式，我可以得出x和y之间的关系。

42x = 28y两边都除以14，得到：3x = 2y

所以，y = (3/2)x

好了，现在我知道小刀的价格是叉子价格的1.5倍。

现在，我想要买若干套餐具，每套包括一把叉子、一个勺子和一个小刀，而且要把钱正好花完。

设我买n套，那么总成本是：n(x + x + y) = n(2x + y)

但是，y = (3/2)x，所以代入得：n(2x + (3/2)x) = n((7/2)x)

所以，总成本是n*(7/2)x

而我的总预算是42x，所以：n*(7/2)x = 42x

两边都除以x（假设x不为零），得到：n*(7/2) = 42

然后，解n：n = 42 / (7/2) = 42 * (2/7) = 12

所以，我可以买12套餐具，每套包含一把叉子、一个勺子和一个小刀，而且把钱正好花完。

但是，我感觉这个答案有点简单，可能哪里想错了。让我再检查一下。

首先，根据题意，钱可以买21叉子和21勺子，或者28小刀。

我设叉子和勺子各为x，小刀为y，然后得到42x = 28y，从而y = (3/2)x

然后，每套餐具是x + x + y = 2x + (3/2)x = (7/2)x

总预算42x，除以每套的成本(7/2)x，得到12套。看起来逻辑是对的。

但是，也许还有其他考虑。

比如，是否必须买整数套？

还是可以单独购买餐具？

题目说要配成一套，并且把钱刚好花完。

所以，应该是买整数套，而且钱要正好花完。

我的计算中，n=12，是整数，符合要求。

或者，可能有其他组合，比如不完全配成套，但题目明确说要配成一套，并且把钱花完，所以应该按照套来买。

所以，我认为答案是买12套餐具，每套包括一把叉子、一个勺子和一个小刀。

最终答案

12