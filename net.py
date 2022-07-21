class Net():
    '''
    This is the base class of Neural Network. Write your own net by implmenting the functions below.
    '''
    def __init__(self, game, args):
        self.game = game
        self.args = args

    @classmethod
    def from_net(cls, net):
        pass
    
    def train(self, trainExamples):
        pass

    def predict(self, canonicalBoard):
        pass

    def load_weights(self, weightsPath):
        pass

    def save_model(self, modelPath):
        pass

    def load_model(self, modelPath):
        pass