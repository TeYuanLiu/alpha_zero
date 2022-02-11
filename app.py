import logging
from bot import Bot
from tictactoe import TicTacToe
from tictactoenet import TicTacToeNet
from utils import *
import numpy as np
from flask import Flask
from flask_cors import CORS
from os import environ

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def factory():    
    args = dotdict({
        'numEpochs': 4,                 # number of epochs
        'numIterations': 20,            # number of self-played games per epoch
        'numMCTSRuns': 40,              # number of MCTS runs per move in game
        'numBattles': 20,               # number of games for bot replacement
        'replaceNetThreshold': 0.55,    # win rate threshold for bot replacement        
        'cpuct': 1,                     # upper confidence bound exploration hyperparameter 
        'maxNumEpochTrainExamples': 4,  # number of epochs recorded
        'tempThreshold': 4,             # number of exploration moves before picking the move with the maximum simulation counts in game
    })

    netArgs = dotdict({
        'lr': 0.001,
        'numEpochs': 100,
        'batchSize': 64,
    })

    game = TicTacToe()
    net = TicTacToeNet(game, netArgs)   
    bot = Bot(game, net, args)
    botDirectoryPath = "./experiments/2021-12-31-15:54:17/bot3"
    bot.load(botDirectoryPath)
    

    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def welcome():
        return f'Welcome to the Alpha TicTacToe Zero API'

    @app.route("/api/<query>")
    def handle_query(query=None):
        board = np.zeros((3, 3))
        if not (query and len(query) == 9): return f'query length is not valid'
        for r in range(3):
            for c in range(3):
                d = query[3*r + c]
                if d == 'O': board[r][c] = 1
                elif d == 'X': board[r][c] = -1
                elif d == '-': continue
                else: return f'query symbol at index {r * 3 + c} is not valid'
        if game.get_outcome(board, 1) != None: return f'game has ended'
        canonicalBoard = game.get_canonical_board(board, 1)
        pi = bot.get_pi(canonicalBoard, temp=0)
        a = np.random.choice(len(pi), p=pi)
        return f'{a}'

    return app
    
if __name__ == "__main__":
    app = factory()
    app.run(host='127.0.0.1', port=environ.get('PORT', 5000))