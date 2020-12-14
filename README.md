# AlphaZero-Othello

An implementation of the AlphaZero algorithm for playing Othello (aka. Reversi)

<img width=500 src="https://raw.githubusercontent.com/PythonNut/alphazero-othello/main/figures/az_iagno_first_win.png"/>
Figure 1: The final board after AlphaZero-Othello beat Iagno "Medium" for the first time!

## What is Othello?

Othello is an abstract strategy board game for two players. 
It is played on an 8×8 board with 64 pieces, called disks, which have different colors on each side (ususally black and white).
The players alternate placing disks with their assigned color facing up.
If a contiguous line of a player's disks along one of the eight directions becomes surrounded on both ends by those of their opponent, all of the middle disks are flipped to the opponent's color.
Every move must produce at least one of these flips.
A player must pass if and only if they have no valid moves and the game ends when both players pass.
The winner is the player with the most disks of their color on the final board. 

On the strength of the best known computer programs relative to humans, [Wikipedia](https://en.wikipedia.org/wiki/Reversi) has this to say:

> Good Othello computer programs play very strongly against human opponents. This is mostly due to difficulties in human look-ahead peculiar to Othello: The interchangeability of the disks and therefore apparent strategic meaninglessness (as opposed to chess pieces for example) makes an evaluation of different moves much harder. 

## What is AlphaZero?

AlphaZero is a computer program created by DeepMind that plays Chess, Shogi, and Go descended from AlphaGo program which famously became the first to achieve superhuman performance playing Go.
Notably, AlphaZero learns to play these games tabula rasa⁠ (i.e. given only the rules of the game itself) and is able to achieve superhuman performance in all three.
The AlphaZero algorithm is easily adapted to other games.

### Main ideas
AlphaZero uses a neural network which takes in the state `s` of the board and outputs two things: `p(s)`, a distribution over set of actions, and `v(s)`, an prediction of which player will win the game. The network is trained to minimize the following loss

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\bg_white&space;L(\theta)&space;=&space;\sum_t&space;\left(&space;\left(v_\theta(s_t)&space;-&space;z_t\right&space;)^2&space;-&space;\hat{\pi}(s_t)^T\log\left(p_\theta(s_t)\right)&space;\right)" target="_blank"><img width=300 src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\bg_white&space;L(\theta)&space;=&space;\sum_t&space;\left(&space;\left(v_\theta(s_t)&space;-&space;z_t\right&space;)^2&space;-&space;\hat{\pi}(s_t)^T\log\left(p_\theta(s_t)\right)&space;\right)" title="L(\theta) = \sum_t \left( \left(v_\theta(s_t) - z_t\right )^2 - \hat{\pi}(s_t)^T\log\left(p_\theta(s_t)\right) \right)" /></a>

where `zₜ` is the outcome of the game (from perspective of time `t`) and `̂π(s)` is an improved policy.
`zₜ` is easy to calculate (we just look at who won in the end) but the computation of `̂π(s)` is more involved.

In order to calculate `̂π(s)` we use Monte Carlo Tree Search (MCTS).
To explain how this works, define

* `Q(s, a)` is the average reward after following action `a` from state `s`.
* `N(s, a)` is the number of times action `a` was chosen at state `s`.
* `P(s, a)` is the probability of taking action `a` from state `s` (according to `p(s)`)

These quantities (which are just implemented using hash tables in practice) are used to calculate

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\bg_white&space;U(s,&space;a)&space;=&space;Q(s,&space;a)&space;&plus;&space;c_{\mathrm{puct}}&space;P(s,&space;a)\frac{\sqrt{\sum_b&space;N(s,&space;b)}}{1&space;&plus;&space;N(s,&space;a)}" target="_blank"><img width=300 src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\bg_white&space;U(s,&space;a)&space;=&space;Q(s,&space;a)&space;&plus;&space;c_{\mathrm{puct}}&space;P(s,&space;a)\frac{\sqrt{\sum_b&space;N(s,&space;b)}}{1&space;&plus;&space;N(s,&space;a)}" title="U(s, a) = Q(s, a) + c_{\mathrm{puct}} P(s, a)\frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}" /></a>

which is called the upper confidence bound of `Q(s, a)` (here, `c_puct` is a hyperparameter that controls how much exploration is done.
During the search, the `a` that maximises `U(s, a)` is chosen.
This is done recursively until the game ends and then the neccesary updates to `Q` and `N` are done at each step back up the call chain.
The search is repeated from the root node many times.
As the number of simulations increases, `Q(s, a)` becomes more accurate and the `U(s, a)` also approach `Q(s, a)`.
After all of the simulations are complete, we assign `N(s, a)/sum(N(s, b) for all b)` to `̂π(s, a)` and the `̂π(s)` and `zₜ` are used to train the network, producing an improved policy and value network for the next iteration. 

## What is AlphaZero-Othello?

AlphaZero-Othello is an implementation of the AlphaZero algorithm that learns to play Othello.
It is written in pure Python, using the PyTorch library to accelerate numerical computations.
The goal was to write the simplest and most readable implementation possible.

* 100% of the code is written by me
* Multithreaded self-play
* Multithreaded evaluation arena
* Uses a single GPU on a single node (i.e. it is not distributed)
* Self-play, evaluation, and training all happen synchronously (unlike in the original AlphaZero)

### Network architecture
The policy and value network is build using residual blocks of the following form
```python
Sequential(
    SumModule(
        Sequential(
            Conv2d(n, k, 3, 1, 1),
            BatchNorm2d(k),
            ReLU(),
            Conv2d(k, k, 3, 1, 1),
            BatchNorm2d(k),
            ReLU(),
        ),
        Conv2d(n, k, 1, 1, 0),
    ),
    ReLU(),
)
```
where `n` of one block equals the `k` of the previous block and the second branch of the `SumModule` is replaced with the identity if `n == k`.
Five blocks are used and the channels numbers are `[16, 32, 64, 128, 128, 128]`.

The output of the residual tower is then split.
The branch that computes `pi` is
```python
Sequential(
    Conv2d(128, 16, 1, 1, 0),
    BatchNorm2d(16),
    Flatten(),
    Linear(16 * 8 * 8, 256),
    BatchNorm1d(256),
    Linear(256, 8 * 8 + 1),
    LogSoftmax(dim=1),
)
```
similarly, the branch that computes `v` is
```python
Sequential(
    Conv2d(128, 16, 1, 1, 0),
    BatchNorm2d(16),
    Flatten(),
    Linear(16 * 8 * 8, 256),
    BatchNorm1d(256),
    Linear(256, 1),
    Sigmoid(),
)
```
### Parameters

Every round of self-play consists of 100 games each moved is based on 25 MCTS simulations. 
20 iterations of training data are preserved in the history buffer.
The `cpuct` parameter is set at `3` and at the root of the MCTS search, Dirchlet noise with `alpha = 0.9` is mixed with estimates of `pi` (25% noise).

### Results

I have played many games against the trained agent and I have never won.
The agent reliably beats the Iagno engine on its "Medium" setting but cannot beat Iagno on its "Hard" setting.

Although the results are not spectacular, they are understandable.
AlphaGo Zero was trained on 4.9 millions games of self play with 1600 simulations per MCTS while this agent was trained on about 40 thousand games with 25 simulations per MCTS.

## Demo

Upload `demo.ipynb` to Google Colab and play against the trained agent!

## References
* [AlphaGo Zero paper](https://www.nature.com/articles/nature24270)
* [AlphaZero paper](http://arxiv.org/abs/1712.01815)
* [MuZero paper](http://arxiv.org/abs/1911.08265)
* [jonathan-laurent/AlphaZero.jl](https://github.com/jonathan-laurent/AlphaZero.jl)
* [This post](https://web.stanford.edu/~surag/posts/alphazero.html) by Surag Nair as well as [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
