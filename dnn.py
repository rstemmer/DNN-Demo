#!/usr/bin/env python3
# dnn, a minimal deep neural network example using tensorflow
# Copyright (C) 2017  Ralf Stemmer <ralf.stemmer@gmx.net>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Do not show warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ID of GPU that shall be used
import tflearn
import tensorflow
from tflearn.layers.core        import input_data, fully_connected
from tflearn.layers.estimator   import regression
from show                       import ShowNeurons, ShowPrediction


class DeepNeuralNetwork(object):
    def __init__(self, usegpu=False):
        self.usegpu    = usegpu
        self.modelfile = "./data/DNN.tfl"



    def LoadModel(self):
        print("\033[1;34mLoading model … \033[0m")
        
        if os.path.exists(self.modelfile+".index"):
            self.model.load(self.modelfile)
        else:
            raise FileNotFoundError("No model data available")

        ShowNeurons("Hidden Layer", self.model, self.hiddenlayer)
        ShowNeurons("Output Layer", self.model, self.outputlayer)



    def SaveModel(self):
        print("\033[1;34mStoreing model … \033[0m")

        if self.model:
            self.model.save(self.modelfile)



    def CreateModel(self):
        print("\033[1;34mCreating model … \033[0m")

        # Reset tensorflow
        tensorflow.reset_default_graph()
        tflearn.config.init_graph()

        # Select device
        if self.usegpu:
            device = "/gpu:0"
        else:
            device = "/cpu:0"

        # Create Model
        with tensorflow.device(device):
            # Input  layer with 2 neurons
            self.inputlayer  = input_data(shape=[None, 2])

            # Hidden layer with 2 neurons
            self.hiddenlayer = fully_connected(
                    self.inputlayer,
                    n_units    = 2,
                    activation = "tanh")
            
            # Output layer with 1 neuron
            self.outputlayer = fully_connected(
                    self.hiddenlayer,
                    n_units    = 1,
                    activation = "sigmoid")

            # Regression layer for training
            reg = regression(self.outputlayer, 
                    optimizer     = "adam", 
                    loss          = "mean_square",
                    learning_rate = 0.1)

            self.model = tflearn.DNN(reg)



    def TrainModel(self, featureset, outputset):
        print("\033[1;34mRunning training process … \033[0m")

        self.model.fit(featureset, outputset, n_epoch=1000)

        ShowNeurons("Hidden Layer", self.model, self.hiddenlayer)
        ShowNeurons("Output Layer", self.model, self.outputlayer)



    def Predict(self, featureset):
        print("\033[1;34mRunning prediction process … \033[0m")
        
        prediction = self.model.predict(featureset)

        return prediction



if __name__ == "__main__":
    # Data for training
    featureset = [[0,0], [0,1], [1,0], [1,1]]
    outputset  = [ [0],   [1],   [1],   [0] ]

    # Create the DNN
    dnn = DeepNeuralNetwork(usegpu=False)
    dnn.CreateModel()

    # Train (l) or predict (p) depending on user input.
    if os.sys.argv[1] == "l":
        dnn.TrainModel(featureset, outputset)
        dnn.SaveModel()

    elif os.sys.argv[1] == "p":
        dnn.LoadModel()
        prediction = dnn.Predict(featureset)
        ShowPrediction(featureset, prediction, outputset)


# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

