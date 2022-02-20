import logging
from bot import Bot
from tictactoe import TicTacToe
from tictactoenet import TicTacToeNet
from collections import deque
from utils import *
import numpy as np
import random

import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import os

random.seed(1)
np.random.seed(1)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Incubator():
    def __init__(self, bot, args):
        self.bot = bot
        self.game = self.bot.game
        self.net = self.bot.net
        self.args = args
    
    def train(self):
        # create experiment directory
        experimentDirectoryPath = os.path.join('./experiments', datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        Path(experimentDirectoryPath).mkdir(parents=True, exist_ok=True)

        trainLosses = []
        trainExamples = deque()
        for i in range(self.args.numEpochs):
            logger.info(f'=== epoch idx: {i} ===')    

            # create bot directory
            botDirectoryPath = os.path.join(experimentDirectoryPath, 'bot' + str(i))
            Path(botDirectoryPath).mkdir(parents=True, exist_ok=True)                  

            # generate play data
            epochTrainExamples = []
            for j in range(self.args.numIterations):
                logger.info(f'= iteration idx: {j} =')
                epochTrainExamples.extend(self.bot.self_play())
            trainExamples.append(epochTrainExamples)
            assert self.args.maxNumEpochTrainExamples >= 1
            if len(trainExamples) > self.args.maxNumEpochTrainExamples:
                trainExamples.popleft()
                assert len(trainExamples) <= self.args.maxNumEpochTrainExamples
            logger.info(f'using {len(trainExamples)} epoch examples')

            # create new bot and train it with data 
            newNet = TicTacToeNet.from_net(self.net)
            epochTrainLosses = newNet.train(trainExamples)
            trainLosses.extend(epochTrainLosses)
            logger.info(f'create new bot')
            newBot = Bot(self.game, newNet, self.args)

            # let new bot fight against current bot and replace it if qualified
            cwins, nwins, draws = self.battle(newBot)
            if not cwins + nwins == 0: logger.info(f'new bot win rate: {nwins / (cwins + nwins)}')
            if cwins + nwins == 0 or nwins / (cwins + nwins) < self.args.replaceNetThreshold:
                logger.info('reject new bot')
            else:                
                logger.info(f'accept new bot')
                self.bot = newBot
            
            self.bot.save(botDirectoryPath)
        
        steps = np.arange(len(trainLosses))
        losses = np.array(trainLosses)
        fig, ax = plt.subplots()
        ax.plot(steps, losses)
        ax.set(xlabel='training steps', ylabel='training loss', title='loss versus step')
        ax.grid()
        plt.savefig(os.path.join(experimentDirectoryPath, "trainLosses.jpg"))

    def battle(self, newBot=None):
        cwins = 0
        nwins = 0
        draws = 0
        numBattles = self.args.numBattles if newBot != None else 1
        for i in range(numBattles):
            logger.debug(f'===== battle {i} start =====')
            for j in [1, -1]:
                board = self.game.get_initial_board()
                player = 1 * j
                
                if newBot == None:
                    logger.debug(f'=== player 1 (O): bot ===')
                    logger.debug(f'=== player -1 (X): you, the challenger ===')
                
                logger.debug(f'=== player {player} goes first ===')

                if newBot == None: self.game.display_board(board)

                while True:
                    logger.debug(f'=== player {player}\'s turn ===')
                    canonicalBoard = self.game.get_canonical_board(board, player)
                    if player == 1: pi = self.bot.get_pi(canonicalBoard, temp=0) 
                    else: 
                        if newBot: pi = newBot.get_pi(canonicalBoard, temp=0)
                        else:
                            userInput = -1
                            while not (0 <= userInput < self.game.get_action_size() and self.game.get_valid_actions(canonicalBoard)[userInput] == 1):
                                userInput = int(input(f'please input valid action (0-{self.game.get_action_size() - 1}): '))
                            pi = [1 if piIdx == userInput else 0 for piIdx in range(self.game.get_action_size())]
                    
                    action = np.random.choice(len(pi), p=pi)
                    logger.debug(f'action: {action}')
                    board = self.game.get_next_board(board, player, action)  

                    if newBot == None: self.game.display_board(board)              

                    outcome = self.game.get_outcome(board)
                    if outcome != None:
                        logger.debug(f'=== outcome ===')
                        if outcome == 1: 
                            cwins += 1
                            logger.debug(f'player {outcome} win')
                        elif outcome == -1: 
                            nwins += 1
                            logger.debug(f'player {outcome} win')
                        elif outcome == 0: 
                            draws += 1
                            logger.debug(f'draw')
                        else:
                            logger.debug(f'unknown game outcome')
                            raise ValueError(f'unknown game outcome')
                        break
                    else:
                        player *= -1
        return (cwins, nwins, draws)

def unit_test():
    args = dotdict({
        'numEpochs': 3,
        'numIterations': 10,
        'numMCTSRuns': 5,
        'numBattles': 20,
        'replaceNetThreshold': 0.55,
        'cpuct': 1,
        'maxNumEpochTrainExamples': 3,
        'tempThreshold': 4,
    })

    netArgs = dotdict({
        'lr': 0.001,
        'numEpochs': 100,
        'batchSize': 64,
    })

    game = TicTacToe()
    net = TicTacToeNet(game, netArgs)    
    bot = Bot(game, net, args)
    
    incubator = Incubator(bot, args)
    incubator.train()
    incubator.battle()
    
    print('======= UNIT TEST END =======') 

if __name__ == "__main__":
    unit_test()