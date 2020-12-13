# AlphaZero-Othello

An implementation of the AlphaZero algorithm for playing Othello (aka. Reversi)

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

## What is AlphaZero-Othello?

AlphaZero-Othello is an implementation of the AlphaZero algorithm that learns to play Othello.
It is written in pure Python, using the PyTorch library to accelerate numerical computations.
The goal was to write the simplest and most readable implementation possible.

* 100% of the code is written by me
* Multithreaded self-play
* Multithreaded evaluation arena
* Uses a single GPU on a single node (i.e. it is not distributed)
* Self-play, evaluation, and training all happen synchronously (unlike in the original AlphaZero)

References:
* [AlphaGo Zero paper](https://www.nature.com/articles/nature24270)
* [AlphaZero paper](http://arxiv.org/abs/1712.01815)
* [MuZero paper](http://arxiv.org/abs/1911.08265)
* [jonathan-laurent/AlphaZero.jl](https://github.com/jonathan-laurent/AlphaZero.jl)
* [This post](https://web.stanford.edu/~surag/posts/alphazero.html) by Surag Nair as well as [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
