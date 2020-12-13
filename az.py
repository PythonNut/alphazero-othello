import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import itertools as it
from collections import defaultdict, deque
from tqdm import trange, tqdm

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class Board(object):
    DIRECTIONS = list(it.product(*[(-1, 0, 1)] * 2))
    DIRECTIONS.remove((0, 0))

    def __init__(self, n=8, board=None):
        if board is None:
            self.n = n
            self.board = torch.zeros((n, n), dtype=torch.int8)
            self.board[n // 2 - 0, n // 2 - 0] = -1
            self.board[n // 2 - 0, n // 2 - 1] = +1
            self.board[n // 2 - 1, n // 2 - 0] = +1
            self.board[n // 2 - 1, n // 2 - 1] = -1
        else:
            self.board = board
            self.n = len(board)

    def copy(self):
        return Board(board=torch.clone(self.board))

    def display(self, axes=False):
        if axes:
            plain = self.display().split("\n")
            top = " ".join(map(str, range(1, self.n + 1)))
            left = [" "] + [chr(97 + i) for i in range(self.n)]
            return "\n".join([f"{a} {b}" for a, b in zip(left, [top] + plain)])

        result = []

        def char(i):
            if i == 0:
                return "·"
            if i == -1:
                return "o"
            if i == 1:
                return "x"

        return "\n".join(" ".join(char(c) for c in row) for row in self.board)

    __repr__ = __str__ = display

    def rep(self):
        return str(self).replace(" ", "").replace("\n", "|")

    def is_pos_inbounds(self, x, y):
        return 0 <= x < self.n and 0 <= y < self.n

    def _extend_in_dir(self, pos, d, n):
        return pos[0] + d[0] * n, pos[1] + d[1] * n

    def is_move_legal(self, x, y, p=1):
        if not self.is_pos_inbounds(x, y):
            return False

        if self.board[x, y] != 0:
            return False

        for d in self.DIRECTIONS:
            for n in range(1, 8):
                xp, yp = self._extend_in_dir((x, y), d, n)

                if not self.is_pos_inbounds(xp, yp):
                    break
                if self.board[xp, yp] == 0 or (n == 1 and self.board[xp, yp] != -p):
                    break
                if n > 1 and self.board[xp, yp] == p:
                    return True

    def all_legal_moves(self, p=1):
        ind = torch.zeros((self.n, self.n), dtype=torch.uint8)
        for i, j in it.product(*[range(self.n)] * 2):
            if self.is_move_legal(i, j, p):
                ind[i, j] = 1
        return ind

    def pi_mask(self, p=1):
        moves = self.all_legal_moves(p)
        may_pass = moves.sum().item() == 0
        return torch.cat([moves.flatten(), torch.tensor([may_pass])])

    def play(self, x, y, p=1):
        assert self.is_move_legal(x, y, p)

        self.board[x, y] = p

        for d in self.DIRECTIONS:
            for n in range(1, 8):
                xp, yp = self._extend_in_dir((x, y), d, n)
                if not self.is_pos_inbounds(xp, yp):
                    break
                if self.board[xp, yp] == 0 or (n == 1 and self.board[xp, yp] != -p):
                    break
                if n > 1 and self.board[xp, yp] == p:
                    for k in range(1, n + 1):
                        xp, yp = self._extend_in_dir((x, y), d, k)
                        self.board[xp, yp] = p
                    break

    def move(self, i, p=1):
        assert 0 <= i < self.n ** 2 + 1
        if i == self.n ** 2:
            return

        x, y = divmod(i, self.n)
        self.play(x, y, p)

    def flip(self):
        self.board *= -1

    def has_moves(self, p=1):
        ind = torch.zeros((self.n, self.n))
        for i, j in it.product(*[range(self.n)] * 2):
            if self.is_move_legal(i, j, p):
                return True
        return False

    def reward(self, p=1):
        if p * self.board.sum() > 0:
            return 1
        elif p * self.board.sum() < 0:
            return -1
        elif self.board.sum() == 0:
            nonzero = float(torch.rand(1) - 0.5) / 100
            assert nonzero != 0
            return nonzero

    def get_game_ended(self, p=1):
        if self.has_moves(p):
            return 0
        if self.has_moves(-p):
            return 0
        return self.reward(p)

    def get_symmetries(self, pi):
        assert len(pi) == self.n ** 2 + 1
        pi_2d = pi[:-1].view(self.n, self.n)
        results = []

        for f, r in it.product(
            (lambda b: b, lambda b: torch.flip(b, [0])),
            map(lambda k: lambda b: torch.rot90(b, k), range(4)),
        ):
            new_self = Board(board=r(f(self.board)))
            new_pi = torch.cat((r(f(pi_2d)).flatten(), pi[-1].ravel()))
            results.append((new_self, new_pi))

        return results


class SumModule(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.branches = nn.ModuleList(modules)

    def forward(self, x):
        return sum(module(x) for module in self.branches)


class Policy(nn.Module):
    def __init__(self, n=8, channels=10):
        super().__init__()
        self.n = n
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)

        self.fc1 = nn.Linear(channels * (n - 2) ** 2, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)

        self.pi_fc = nn.Linear(256, n ** 2 + 1)
        self.v_fc = nn.Linear(256, 1)
        self.eval()

    def forward(self, board):
        x = board
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = F.relu(x)

        pi = self.pi_fc(x)
        v = self.v_fc(x)

        return F.log_softmax(pi, dim=1), torch.sigmoid(v).squeeze(1)

    def predict_board(self, board):
        assert isinstance(board, Board)
        log_pi, v = self.forward(
            board.board.reshape(1, 1, board.n, board.n)
            .float()
            .to(self.conv1.weight.device)
        )
        return log_pi.detach().squeeze().exp().cpu(), v.detach().item()


class PolicyBig(nn.Module):
    def __init__(self, n=8, channels=10):
        assert n == 8
        super().__init__()
        self.n = n

        self.residual_tower = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(16, 32, 3, 1, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                ),
                nn.Conv2d(16, 32, 1, 1, 0),
            ),
            nn.ReLU(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(32, 64, 3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                ),
                nn.Conv2d(32, 64, 1, 1, 0),
            ),
            nn.ReLU(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(64, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ),
                nn.Conv2d(64, 128, 1, 1, 0),
            ),
            nn.ReLU(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ),
                nn.Sequential(),
            ),
            nn.ReLU(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ),
                nn.Sequential(),
            ),
            nn.ReLU(),
        )

        self.pi_tower = nn.Sequential(
            nn.Conv2d(128, 16, 1, 1, 0),
            nn.BatchNorm2d(16),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, n ** 2 + 1),
            nn.LogSoftmax(dim=1),
        )

        self.v_tower = nn.Sequential(
            nn.Conv2d(128, 16, 1, 1, 0),
            nn.BatchNorm2d(16),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.eval()

    def forward(self, board):
        x = self.residual_tower(board)
        pi = self.pi_tower(x)
        v = self.v_tower(x)
        return pi, v

    def predict_board(self, board):
        assert isinstance(board, Board)
        with torch.no_grad():
            log_pi, v = self.forward(
                board.board.reshape(1, 1, board.n, board.n)
                .float()
                .to(self.residual_tower[0].weight.device)
            )
            return log_pi.detach().squeeze().exp().cpu(), v.detach().item()


class SelfPlayDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, i):
        example = self.examples[i]
        return (
            example[0].board.unsqueeze(0).float(),
            example[1],
            torch.tensor(example[2]).float(),
        )

    def __len__(self):
        return len(self.examples)


def make_train_dataloader(dataset, batch_size=64):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


def train(net, examples, epochs=10, batch_size=128):
    torch.multiprocessing.set_start_method("fork", force=True)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    dataloader = make_train_dataloader(SelfPlayDataset(examples), batch_size)
    net.train()
    batch_count = len(examples) // batch_size * epochs
    with tqdm(total=batch_count, desc="train", ncols=80) as pbar:
        for epoch in range(epochs):
            for (boards, target_pis, target_vs) in dataloader:
                opt.zero_grad()
                boards, target_pis, target_vs = (
                    boards.cuda(),
                    target_pis.cuda(),
                    target_vs.cuda(),
                )

                # compute output
                log_pis, vs = net(boards)
                pi_loss = -torch.mean(log_pis * target_pis)
                v_loss = torch.mean((target_vs - vs).pow(2))
                total_loss = pi_loss + v_loss

                total_loss.backward()
                opt.step()
                pbar.update()

    torch.multiprocessing.set_start_method("spawn", force=True)
    net.eval()


class MCTS(object):
    def __init__(self, policy, cpuct=3, numMCTSSims=25):
        self.policy = policy
        self.cpuct = cpuct
        self.numMCTSSims = numMCTSSims

        self.Qsa = defaultdict(lambda: 0)
        self.Nsa = defaultdict(lambda: 0)
        self.Ns = defaultdict(lambda: 0)
        self.Ps = {}

        self.Es = {}
        self.Vs = {}

    def search(self, board, dir_eps=0.25):
        s = board.rep()

        if s not in self.Es:
            self.Es[s] = board.get_game_ended()

        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            self.Ps[s], v = self.policy.predict_board(board)

            if dir_eps > 0:
                alpha = torch.empty_like(self.Ps[s]).fill_(0.9)
                noise = torch.distributions.Dirichlet(alpha).sample()
                self.Ps[s] = (1 - dir_eps) * self.Ps[s] + dir_eps * noise

            valids = board.pi_mask()
            self.Ps[s] = self.Ps[s] * valids
            if self.Ps[s].sum() == 0:
                self.Ps[s] += valids
            self.Ps[s] /= self.Ps[s].sum()
            self.Vs[s] = valids
            return -v

        valids = self.Vs[s]
        best_u, best_a = -float("inf"), None

        for a in range(board.n ** 2 + 1):
            if not valids[a]:
                continue
            u = self.Qsa[s, a] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                1 + self.Nsa[s, a]
            )

            if u > best_u:
                best_u, best_a = u, a

        a = best_a
        next_board = board.copy()
        next_board.move(a)
        next_board.flip()

        v = self.search(next_board, dir_eps=0)

        self.Qsa[s, a] = (self.Nsa[s, a] * self.Qsa[s, a] + v) / (self.Nsa[s, a] + 1)
        self.Nsa[s, a] += 1
        self.Ns[s] += 1

        return -v

    def pi(self, board, temp=1, dir_eps=0.25):
        for i in range(self.numMCTSSims):
            self.search(board, dir_eps)

        s = board.rep()
        weights = torch.tensor([self.Nsa[(s, a)] for a in range(board.n ** 2 + 1)])
        if temp == 0:
            hot = int(
                torch.distributions.Categorical(weights == weights.max()).sample()
            )
            weights = torch.zeros_like(weights)
            weights[hot] = 1
            return weights

        weights = weights ** (1 / temp)
        weights /= weights.sum()
        return weights


class Player(object):
    def __call___(self):
        raise NotImplementedError

    def init(self):
        pass


class MCTSPlayer(Player):
    def __init__(self, policy, dir_eps=0.25, temp=0.2, cpuct=3, numMCTSSims=25):
        self.policy = policy
        self.dir_eps = dir_eps
        self.temp = temp
        self.cpuct = cpuct
        self.numMCTSSims = numMCTSSims
        # self.mcts = None
        # self.mcts = MCTS(self.policy, cpuct=self.cpuct, numMCTSSims=self.numMCTSSims)

    def __call__(self, board):
        pi = self.mcts.pi(board, self.temp, self.dir_eps)
        return int(torch.argmax(pi))

    def init(self):
        self.mcts = MCTS(self.policy, cpuct=self.cpuct, numMCTSSims=self.numMCTSSims)


class HumanPlayer(Player):
    def __call__(self, board):
        print(board.display(axes=True))
        while True:
            move = input("> ")
            if move.startswith("p"):
                move = board.n ** 2
            try:
                r = ord(move[0]) - 97
                c = int(move[1]) - 1
                move = r * board.n + c
            except:
                print("failed to parse move")
                continue

            if not board.pi_mask()[move]:
                print("illegal move")
                continue

            break

        preview = board.copy()
        preview.move(move)
        print(preview.display(axes=True))

        return move


class RandomPlayer(Player):
    def __call__(self, board):
        pi_mask = board.pi_mask()
        hot = int(torch.distributions.Categorical(pi_mask > 0.5).sample())
        return hot


class Arena(object):
    def __init__(self, p1, p2):
        self.p1, self.p2 = p1, p2

    def play(self, first=-1):
        players = [self.p2, None, self.p1]
        p = first
        board = Board()
        while not board.get_game_ended():
            action = players[p + 1](board)
            board.move(action)
            board.flip()
            p *= -1

        torch.cuda.empty_cache()
        return p * board.get_game_ended()

    def play_ntimes(self, n, verbose=False):
        self.p1.init()
        self.p2.init()
        p1wins, p2wins, draws = 0, 0, 0
        flip = 2 * torch.randint(2, (1,)).item() - 1
        for i in range(n):
            result = self.play(first=flip * ((i % 2 * 2) - 1))
            if result == 1:
                p1wins += 1
            elif result == -1:
                p2wins += 1
            else:
                draws += 1

        return p1wins, p2wins, draws

    # def play_ntimes(self, n, verbose=False):
    #     p1wins, p2wins, draws = 0, 0, 0
    #     with torch.multiprocessing.Pool(8) as p:
    #         with tqdm(total=n, desc="arena") as pbar:
    #             for result in p.imap_unordered(
    #                 self.play, [(i % 2 * 2) - 1 for i in range(n)]
    #             ):
    #                 if result == 1:
    #                     p1wins += 1
    #                 elif result == -1:
    #                     p2wins += 1
    #                 else:
    #                     draws += 1
    #                 pbar.update()

    #     return p1wins, p2wins, draws

    def play_nktimes(self, n, k, verbose=False):
        p1wins, p2wins, draws = 0, 0, 0
        with torch.multiprocessing.Pool(8) as p:
            with tqdm(total=k, desc="arena", ncols=80) as pbar:
                for p1ws, p2ws, ds in p.imap_unordered(
                    self.play_ntimes, [n for i in range(k)]
                ):
                    p1wins += p1ws
                    p2wins += p2ws
                    draws += ds
                    pbar.update()

        return p1wins, p2wins, draws


class Coach(object):
    def __init__(
        self,
        checkpoint="checkpoints",
        numIters=1000,
        numEps=100,
        tempThreshold=32,
        updateThreshold=0.55,
        maxlenOfQueue=200000,
        arenaCompare=40,
        numItersForTrainExamplesHistory=20,
    ):
        self.numIters = numIters
        self.numEps = numEps
        self.checkpoint = Path(checkpoint)
        self.tempThreshold = tempThreshold
        self.updateThreshold = updateThreshold
        self.maxlenOfQueue = maxlenOfQueue
        self.arenaCompare = arenaCompare
        self.numItersForTrainExamplesHistory = numItersForTrainExamplesHistory
        self.nnet = PolicyBig().cuda()
        self.pnet = PolicyBig().cuda()
        self.example_hist = deque([], maxlen=numItersForTrainExamplesHistory)
        self.startIter = 1

    # static so we don't drag example_hist into every worker
    @staticmethod
    def episode(data):
        nnet, tempThreshold = data
        examples = []
        board = Board()
        p = 1
        step = 0
        mcts = MCTS(nnet)

        while True:
            step += 1
            temp = 1 if step < tempThreshold else 0.3
            pi = mcts.pi(board, temp)
            sym = board.get_symmetries(pi)
            for sb, sp in sym:
                examples.append((sb, sp, p))

            action = int(torch.distributions.Categorical(pi).sample())

            board.move(action)
            board.flip()
            p *= -1

            r = board.get_game_ended()
            if r != 0:
                sbs = torch.stack([x[0].board for x in examples])
                sps = torch.stack([x[1] for x in examples])
                rs = [r * (-1) ** (x[2] != p) for x in examples]
                torch.cuda.empty_cache()
                return sbs, sps, rs
                # return [(x[0], x[1], r * ((-1) ** (x[2] != p))) for x in examples]

    def learn(self):
        for i in range(self.startIter, self.numIters + 1):
            print(f"########## iter {i}", sum(map(len, self.example_hist)))
            iter_examples = deque([], maxlen=self.maxlenOfQueue)
            if self.startIter == 1 or i > self.startIter:
                with torch.multiprocessing.Pool(8) as p:
                    with tqdm(total=self.numEps, desc="self play", ncols=80) as pbar:
                        for examples in p.imap_unordered(
                            self.episode,
                            [(self.nnet, self.tempThreshold)] * self.numEps,
                        ):
                            iter_examples.extend(
                                (
                                    (Board(board=sb), sp, r)
                                    for sb, sp, r in zip(*examples)
                                )
                            )
                            pbar.update()

                # for _ in trange(self.numEps, desc="self play"):
                #     # iter_examples.extend(self.episode())
                #     iter_examples.extend(((Board(board=sb), sp, r) for sb, sp, r in zip(*self.episode())))

                self.example_hist.append(iter_examples)
                self.save_examples(i - 1)

            examples = []
            for e in self.example_hist:
                examples.extend(e)

            torch.save(self.nnet.state_dict(), self.checkpoint / "temp.pt")
            self.pnet.load_state_dict(torch.load(self.checkpoint / "temp.pt"))
            pmp = MCTSPlayer(self.pnet)

            train(self.nnet, examples)

            nmp = MCTSPlayer(self.nnet)
            arena = Arena(pmp, nmp)
            pwins, nwins, draws = arena.play_nktimes(self.arenaCompare // 8, 8)
            print(f"n/p wins: {nwins} / {pwins} ({draws} draws)")

            if pwins + nwins == 0 or nwins / (pwins + nwins) < self.updateThreshold:
                print("rejected new model")
                self.nnet.load_state_dict(torch.load(self.checkpoint / "temp.pt"))
            else:
                print("accepting new model")
                torch.save(self.nnet.state_dict(), self.checkpoint / f"iter{i:05d}.pt")
                torch.save(self.nnet.state_dict(), self.checkpoint / f"best.pt")

    def save_examples(self, i):
        torch.save(self.example_hist, self.checkpoint / f"iter{i:05d}.examples.pt")

    def load(self, i):
        self.startIter = i + 1
        self.example_hist = torch.load(self.checkpoint / f"iter{i:05d}.examples.pt")
        self.nnet.load_state_dict(torch.load(self.checkpoint / f"iter{i:05d}.pt"))
        self.pnet.load_state_dict(torch.load(self.checkpoint / f"iter{i:05d}.pt"))


# if __name__ == "__main__":
#     P = Policy()
#     B = Board()
#     mcts = MCTS(P)
#     MP = MCTSPlayer(mcts)
#     A = Arena(MP, MP)
#     # A.play_ntimes(2)

# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method("spawn", force=True)
#     torch.multiprocessing.set_sharing_strategy("file_system")
#     C = Coach()
#     # C.load(57)
#     C.learn()

# if __name__ == "__main__":
#     P = Policy().cuda()
#     MP = MCTSPlayer(P, 0, 0, cpuct=3, numMCTSSims=500)
#     MP.init()
#     A = Arena(MP, HumanPlayer())
#     A.play(1)
