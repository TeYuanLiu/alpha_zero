import logging
from urllib import response
from bot import Bot
from tictactoe import TicTacToe
from tictactoenet import TicTacToeNet
from utils import *
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from os import environ

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def factory():    
    argsEasy = dotdict({
        'numMCTSRuns': 5,              # number of MCTS runs per move in game    
        'cpuct': 1,                     # upper confidence bound exploration hyperparameter
    })

    argsHard = dotdict({
        'numMCTSRuns': 20,              # number of MCTS runs per move in game    
        'cpuct': 1,                     # upper confidence bound exploration hyperparameter
    })

    netArgs = dotdict({
        'lr': 0.001,
        'numEpochs': 100,
        'batchSize': 64,
    })

    game = TicTacToe()

    netEasy = TicTacToeNet(game, netArgs)   
    botEasy = Bot(game, netEasy, argsEasy)
    botEasyDirectoryPath = "./experiments/2022-02-19-21:59:00/bot0"
    botEasy.load(botEasyDirectoryPath)

    netHard = TicTacToeNet(game, netArgs)   
    botHard = Bot(game, netHard, argsHard)
    botHardDirectoryPath = "./experiments/2021-12-31-15:54:17/bot3"
    botHard.load(botHardDirectoryPath)

    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def welcome():
        return f'Welcome to the Alpha TicTacToe Zero Server'

    @app.route('/api', methods=["GET", 'POST'])
    def api():
        if request.method == "POST":
            dic = request.get_json()
            if 'botIsO' not in dic or 'easyMode' not in dic or 'boardString' not in dic: return f"request is not valid"
            print(dic)
            botIsO, easyMode, boardString = dic['botIsO'], dic['easyMode'], dic['boardString']
            board = np.zeros((3, 3))
            if not (boardString and len(boardString) == 9): return f'boardString length is not valid'
            for r in range(3):
                for c in range(3):
                    d = boardString[3*r + c]
                    if d == 'o': board[r][c] = 1
                    elif d == 'x': board[r][c] = -1
                    elif d == '-': continue
                    else: return f'boardString value at index {r * 3 + c} is not valid'
            if game.get_outcome(board) != None: return f'game has ended'
            canonicalBoard = game.get_canonical_board(board, 1) if botIsO == "true" else game.get_canonical_board(board, -1)
            pi = botEasy.get_pi(canonicalBoard, temp=0) if easyMode == "true" else botHard.get_pi(canonicalBoard, temp=0)
            a = np.random.choice(len(pi), p=pi)
            return f"{a}", 200
        else:
            message = {"greeting": "Hello from the API"}
            return jsonify(message)

    return app
    
if __name__ == "__main__":
    app = factory()
    app.run(host='127.0.0.1', port=environ.get('PORT', 5000))