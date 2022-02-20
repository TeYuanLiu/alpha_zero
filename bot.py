import logging
from tictactoe import TicTacToe
from tictactoenet import TicTacToeNet
from math import sqrt
import numpy as np
from utils import *
import os
import pickle

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EPSILON = 1e-8

np.random.seed(1)

class Bot():
    def __init__(self, game, net, args):
        self.game = game
        self.net = net
        self.args = args

        self.Qsa = {} # q value of state s action a
        self.Ps = {} # action probability vector of state s
        self.Nsa = {} # visit count of state s action a
        self.Ns = {} # visit count of state s
        self.Es = {} # game outcome of state s
        self.Vs = {} # valid action vector of state s
        
    def self_play(self):
        iterationTrainExamples = []
        board = self.game.get_initial_board()
        player = 1
        step = 0

        while True:
            step += 1
            logger.debug(f'======= player {player}\'s turn =======')
            canonicalBoard = self.game.get_canonical_board(board, player)
            temp = int(step < self.args.tempThreshold)
            pi = self.get_pi(canonicalBoard, temp)
            logger.debug(f'pi: {pi}')

            syms = self.game.get_symmetries(canonicalBoard, pi)

            iterationTrainExamples.extend([[sym[0], player, sym[1]] for sym in syms])
            
            action = np.random.choice(len(pi), p=pi)
            logger.debug(f'action: {action}')
            board = self.game.get_next_board(board, player, action)

            outcome = self.game.get_outcome(board)
            if outcome != None:
                logger.debug(f'======= outcome =======')
                if outcome == 1: logger.debug(f'player {outcome} win')
                elif outcome == -1: logger.debug(f'player {outcome} win')
                elif outcome == 0: logger.debug(f'draw')
                return [[example[0], example[2], outcome * example[1]] for example in iterationTrainExamples] # discourage draw plays?
            else:
                player *= -1

    def get_pi(self, canonicalBoard, temp=1):
        s = self.game.get_board_string(canonicalBoard)
        for _ in range(self.args.numMCTSRuns):
            logger.debug(f'======= MCTS =======')
            canonicalBoardMCTSRun = self.game.get_canonical_board(canonicalBoard, 1)
            self.search(canonicalBoardMCTSRun, s)

        numActionVisits = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]
        numStateVisits = sum(numActionVisits)
        if temp: pi = [n / numStateVisits for n in numActionVisits]
        else: 
            maxCountIdx = numActionVisits.index(max(numActionVisits))
            pi = [1 if idx == maxCountIdx else 0 for idx, val in enumerate(numActionVisits)]
        return pi
    
    def search(self, canonicalBoard, s):
        logger.debug(f'===== search =====')
        if s not in self.Es:
            self.Es[s] = self.game.get_outcome(canonicalBoard)
        
        if self.Es[s] != None:
            logger.debug(f'=== leaf backup ===')
            logger.debug(f'outcome v: {-self.Es[s]}')
            return -self.Es[s]
        
        if s not in self.Ps:
            logger.debug(f'=== branch backup ===')
            self.Ps[s], v = self.net.predict(canonicalBoard)
            logger.debug(f'Ps[s]: {self.Ps[s]}\nv: {v}')
            validActions = self.game.get_valid_actions(canonicalBoard)
            logger.debug(f'validActions: {validActions}')
            self.Ps[s] = np.multiply(self.Ps[s], validActions)
            logger.debug(f'Ps[s]: {self.Ps[s]}')
            sumPsS = np.sum(self.Ps[s])
            if sumPsS > 0:
                self.Ps[s] = self.Ps[s] / sumPsS
            else:
                logger.warning(f'no valid actions found! use a work around.')
                self.Ps[s] = self.Ps[s] + validActions
                self.Ps[s] = self.Ps[s] / np.sum(self.Ps[s])
            
            self.Vs[s] = validActions
            self.Ns[s] = 0

            return -v
        
        logger.debug(f'=== select ===')
        validActions = self.Vs[s]
        logger.debug(f'validActions: {validActions}')
        maxBound = -float('inf')
        bestAction = -1

        for a in range(self.game.get_action_size()):
            if validActions[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * sqrt(self.Ns[s] + EPSILON)
                
                if u > maxBound:
                    maxBound = u
                    bestAction = a
        
        a = bestAction
        logger.debug(f'action: {a}')
        nextPlayerCanonicalBoard = -1 * self.game.get_next_board(canonicalBoard, 1, a)
        nextS = self.game.get_board_string(nextPlayerCanonicalBoard)

        logger.debug(f'=== expand ===')
        v = self.search(nextPlayerCanonicalBoard, nextS)

        logger.debug(f'=== backup ===')
        if (s, a) in self.Qsa and (s, a) in self.Nsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        self.Ns[s] += 1
        return -v

    def save(self, botDirectoryPath):
        modelPath = os.path.join(botDirectoryPath, 'model')
        self.net.save_model(modelPath)

        mctsPath = os.path.join(botDirectoryPath, 'mcts.p')
        mcts = {'Qsa': self.Qsa, 'Ps': self.Ps, 'Nsa': self.Nsa, 'Ns': self.Ns, 'Es': self.Es, 'Vs': self.Vs}
        with open(mctsPath, 'wb') as f:
            pickle.dump(mcts, f)

    def load(self, botDirectoryPath):
        modelPath = os.path.join(botDirectoryPath, 'model')
        self.net.load_model(modelPath)

        mctsPath = os.path.join(botDirectoryPath, 'mcts.p')
        with open(mctsPath, 'rb') as f:
            mcts = pickle.load(f)
            self.Qsa = mcts['Qsa']
            self.Ps = mcts['Ps']
            self.Nsa = mcts['Nsa']
            self.Ns = mcts['Ns']
            self.Es = mcts['Es']
            self.Vs = mcts['Vs']

def unit_test():
    print('======= UNIT TEST START =======')
    args = dotdict({
        'numEpochs': 4,
        'numIterations': 20,
        'replaceNetThreshold': 0.55,
        'numMCTSRuns': 40,
        'numBattles': 20,
        'cpuct': 1,
        'tempThreshold': 4,
    })

    netArgs = dotdict({
        'lr': 0.001,
        'numEpochs': 10,
        'batchSize': 64,
    })

    game = TicTacToe()
    net = TicTacToeNet(game, netArgs)
    bot = Bot(game, net, args)
    
    botDirectoryPath = "./experiments/2021-12-31-15:54:17/bot3"
    bot.load(botDirectoryPath)
    
    bot.self_play()

    print('======= UNIT TEST END =======') 

if __name__ == "__main__":
    unit_test()