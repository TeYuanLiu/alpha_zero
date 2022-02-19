import logging
from incubator import Incubator
from bot import Bot
from tictactoe import TicTacToe
from tictactoenet import TicTacToeNet
from utils import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():

    # reinforcement learning and MCTS parameters
    args = dotdict({
        'numEpochs': 3,                 # number of epochs
        'numIterations': 20,            # number of self-played games per epoch
        'numMCTSRuns': 10,              # number of MCTS runs per move in game
        'numBattles': 20,               # number of games for bot replacement
        'replaceNetThreshold': 0.55,    # win rate threshold for bot replacement        
        'cpuct': 1,                     # upper confidence bound exploration hyperparameter 
        'maxNumEpochTrainExamples': 3,  # number of epochs recorded
        'tempThreshold': 4,             # number of exploration moves before picking the move with the maximum simulation counts in game
    })

    # neural network parameters
    netArgs = dotdict({
        'lr': 0.001,                    # neural network learning rate
        'numEpochs': 100,               # number of epochs
        'batchSize': 64,                # batch size
    })

    game = TicTacToe()
    net = TicTacToeNet(game, netArgs)
    bot = Bot(game, net, args)

    incubator = Incubator(bot, args)
    incubator.train()
    incubator.battle()

if __name__ == "__main__":
    main()