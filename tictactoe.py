from game import Game
import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TicTacToe(Game):
    def __init__(self, n=3):
        self.n = n

    def get_initial_board(self):
        return np.zeros((self.n, self.n), dtype=int)                 # note the data type

    def get_canonical_board(self, board, player):
        return player * np.copy(board)

    def get_next_board(self, board, player, action):
        if 0 <= action < self.get_action_size():
            rowIdx = action // self.n
            colIdx = action % self.n
            assert board[rowIdx, colIdx] == 0
            board[rowIdx, colIdx] = player
        return board

    def get_outcome(self, board):
        for i in range(self.n):
            if np.sum(board[i, :]) == self.n or np.sum(board[:, i]) == self.n:
                return 1
            elif np.sum(board[i, :]) == -self.n or np.sum(board[:, i]) == -self.n:
                return -1
            
        if np.trace(board) == self.n or np.trace(board[:, ::-1]) == self.n:
            return 1
        elif np.trace(board) == -self.n or np.trace(board[:, ::-1]) == -self.n:
            return -1
        
        if np.count_nonzero(board) == self.n * self.n: return 0
        
        return None

    def get_board_string(self, canonicalBoard):
        return np.array2string(canonicalBoard.astype(int))               # note the data type         

    def get_valid_actions(self, canonicalBoard):
        validActions = np.zeros(self.get_action_size(), dtype=int)               # note the data type
        for indices in np.argwhere(canonicalBoard == 0):
            validActions[self.n * indices[0] + indices[1]] = 1
        return validActions

    def get_action_size(self):
        return self.n * self.n

    def display_board(self, board, symbol=True):
        print('=============')
        print('=   BOARD   =')
        print('=============')
        for i in range(self.n):
            if symbol: rowDisk = ['O' if board[i][j] == 1 else 'X' if board[i][j] == -1 else ' ' for j in range(self.n)]
            else: rowDisk = [str(board[i][j]) for j in range(self.n)]
            if i == self.n - 1: rowString = "  |  ".join(rowDisk)
            else: rowString = " _|_ ".join(rowDisk)
            print(rowString)
        print('=============')
        print('=   BOARD   =')
        print('=============') 

    def get_board_shape(self):
        return (self.n, self.n) 

    def get_symmetries(self, canonicalBoard, pi):
        piBoard = np.reshape(pi, (self.n, self.n))
        syms = []
        for flip in [False, True]:
            for rotate in range(4):
                rotatedCanonicalBoard = np.fliplr(canonicalBoard) if flip else canonicalBoard
                rotatedPiBoard = np.fliplr(piBoard) if flip else piBoard

                rotatedCanonicalBoard = np.rot90(rotatedCanonicalBoard, k=rotate)
                rotatedPiBoard = np.rot90(rotatedPiBoard, k=rotate)               

                syms.append([rotatedCanonicalBoard, rotatedPiBoard.ravel()])
        return syms            

def unit_test():
    print('======= UNIT TEST START =======')
    game = TicTacToe()
    
    board = game.get_initial_board()
    game.display_board(board)
    player = 1
    
    while True:
        canonicalBoard = game.get_canonical_board(board, player)
        playerIcon = 'O' if player == 1 else 'X'
        validActions = game.get_valid_actions(canonicalBoard)
        print(f"Available actions for {playerIcon}: {validActions}")
        action = int(input("Enter action: "))
        assert 0 <= action < game.get_action_size() and validActions[action] == 1
        board = game.get_next_board(board, player, action)
        game.display_board(board)
        outcome = game.get_outcome(board)
        if outcome != None:
            if outcome == 1: print("O win")
            elif outcome == -1: print("X win")
            elif outcome == 0: print("draw")
            break
        player *= -1
    
    board = np.array([[1,2,3],[4,5,6],[7,8,9]])
    pi = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    game.display_board(board, False)
    print(pi)
    syms = game.get_symmetries(board, pi)
    for sym in syms:
        print('----------------------------------------')
        game.display_board(sym[0], False)
        print(sym[1])
    print('======= UNIT TEST END =======') 

if __name__ == "__main__":
    unit_test()