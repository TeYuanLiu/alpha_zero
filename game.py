class Game():
    '''
    This is the base class of Game. Write your own game by implmenting the functions below.
    '''
    def __init__(self):
        pass

    def get_initial_board(self):
        pass

    def get_canonical_board(self, board, player):
        pass

    def get_next_board(self, board, player, action):
        pass

    def get_outcome(self, board):
        pass

    def get_board_string(self, canonicalBoard):
        pass

    def get_valid_actions(self, canonicalBoard):
        pass

    def get_action_size(self):
        pass

    def display_board(self, board, symbol):
        pass

    def get_board_shape(self):
        pass

    def get_symmetries(self, canonicalBoard, pi):
        pass