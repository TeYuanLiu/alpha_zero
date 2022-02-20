import tensorflow
from tensorflow.keras import Input
from tensorflow.keras.layers import Reshape, Conv2D, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.models import clone_model, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tictactoe import TicTacToe
from utils import *

import numpy as np
import logging
import os
from pathlib import Path

tensorflow.random.set_seed(1)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TicTacToeNet():
    def __init__(self, game, args, modelWeights=None):
        self.game = game
        self.args = args
        self.boardWidth, self.boardHeight = self.game.get_board_shape()
        self.actionSize = self.game.get_action_size()
        self.model = None
        
        inputs = Input(shape=(self.boardWidth, self.boardHeight))
        reshapes = Reshape((self.boardWidth, self.boardHeight, 1))(inputs)
        conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(filters=81, kernel_size=2, padding='valid')(reshapes)))
        conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(filters=243, kernel_size=2, padding='valid')(conv1)))
        conv2Flat = Flatten()(conv2)
        fc1 = Activation('relu')(BatchNormalization(axis=1)(Dense(128)(conv2Flat)))
        fc2 = Activation('relu')(BatchNormalization(axis=1)(Dense(64)(fc1)))
        outputPi = Dense(units=self.actionSize, activation='softmax', name='pi')(fc2)
        outputV = Dense(units=1, activation='tanh', name='v')(fc2)

        self.model = Model(inputs=inputs, outputs=[outputPi, outputV])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate=self.args.lr))
        
        if modelWeights: self.model.set_weights(modelWeights)
        #self.model.summary()

    @classmethod
    def from_net(cls, net):
        modelWeights = net.model.get_weights()
        return cls(net.game, net.args, modelWeights)       
    
    def train(self, trainExamples):
        logger.info(f'start neural network training')
        canonicalBoards = np.array([example[0] for iterationExamples in trainExamples for example in iterationExamples])
        pis = np.array([example[1] for iterationExamples in trainExamples for example in iterationExamples])
        vs = np.array([example[2] for iterationExamples in trainExamples for example in iterationExamples])
        logger.info(f'canonicalBoards shape: {canonicalBoards.shape}')
        logger.info(f'pis shape: {pis.shape}')
        logger.info(f'vs shape: {vs.shape}')

        checkpointDirectoryPath='./tmp'
        Path(checkpointDirectoryPath).mkdir(parents=True, exist_ok=True)

        checkpointPath = os.path.join(checkpointDirectoryPath, 'weights')
        checkpoint = ModelCheckpoint(filepath=checkpointPath, save_weights_only=True, monitor='loss', verbose=1, mode='min', save_best_only=True)
        earlyStop = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=3)
        history = self.model.fit(x=canonicalBoards, y=[pis, vs], batch_size=self.args.batchSize, epochs=self.args.numEpochs, callbacks=[checkpoint, earlyStop])
        self.load_weights(checkpointPath)

        return history.history['loss']
        
    def predict(self, canonicalBoard):
        prediction = self.model.predict(np.expand_dims(canonicalBoard, axis=0))
        return (prediction[0][0], prediction[1][0][0])

    def load_weights(self, weightsPath):
        self.model.load_weights(weightsPath).expect_partial()

    def save_model(self, modelPath):
        self.model.save(modelPath)

    def load_model(self, modelPath):
        self.model = load_model(modelPath)

def unit_test():
    print('======= UNIT TEST START =======')
    game = TicTacToe()
    args = dotdict({
        'lr': 0.001,
        'numEpochs': 10,
        'batchSize': 64,
    })
    net = TicTacToeNet(game, args)
    
    canonicalBoard = np.array([[1,-1,1],[-1,1,-1],[-1,0,1]])
    pi = np.array([.1, .1, .1, .1, .2, .1, .1, .1, .1])
    v = np.array([0.5])
   
    predP, predV = net.predict(canonicalBoard)
    print(f'predV: {predV}')

    newNet = TicTacToeNet.from_net(net)
    
    predP2, predV2 = newNet.predict(canonicalBoard)
    print(f'cloned predV: {predV2}')

    assert predV == predV2 and predP == predP2
    
    trainExamples = [[[canonicalBoard, pi, v]]]
    newNet.train(trainExamples)
    predP3, predV3 = newNet.predict(canonicalBoard)
    print(f'trained cloned predV: {predV3}') 

    assert predV != predV3 and predP != predP3
    
    print('======= UNIT TEST END =======') 

if __name__ == "__main__":
    unit_test()