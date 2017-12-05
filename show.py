import tflearn


def ShowNeurons(title, model, layer):
    with model.session.as_default():
        rawbiasdata = tflearn.variables.get_value(layer.b)

    rawweightdata = model.get_weights(layer.W)
    numofweights = len(rawweightdata)
    numofneurons = len(rawweightdata[0])

    neurons = []
    for neuronindex in range(numofneurons):
        weightlist = []
        for weightindex in range(numofweights):
            weightlist.append(rawweightdata[weightindex][neuronindex])
        neurons.append(weightlist)
    
    print("\n\033[1;37m{: ^31}".format(title))

    for index, neuron in enumerate(neurons):
        print("\033[1;30mbias  ⮦ weights\033[1;35m┌────┐ \033[0;35mNeuron {: 2d}".format(index))
        print("\033[1;30m ⮡ \033[0;34m(\033[0;36m{: 3.3f}\033[0;34m)───▶\033[1;35m│    │\033[1;30m ⮠\033[0m".format(rawbiasdata[index]))
        for weight in neuron:
            print("\033[1;34m───(\033[1;36m{: 3.3f}\033[1;34m)───▶\033[1;35m│    │\033[0m".format(weight))
        print("\033[1;35m               └────┘")


def ShowPrediction(featureset, predictions, expectedoutputs):
    print("\n\033[1;37m{: ^22}".format("Prediction"))

    print("\033[1;34m┌───┬───┬─────────┬───┐")
    print("\033[1;34m│\033[1;36m x₁\033[1;34m│\033[1;36m x₂\033[1;34m│\033[1;35m    ŷ    \033[1;34m│\033[1;32m y \033[1;34m│")
    print("\033[1;34m├───┼───┼─────────┼───┤")
    for predictionnumber, prediction in enumerate(predictions):
        for feature in featureset[predictionnumber]:
            print("│ \033[1;36m{:1d}\033[1;34m".format(feature), end=" ")
        print("│\033[1;35m  {: 2.3f}\033[1;34m".format(prediction[0]), end=" ")
        print("│\033[1;32m {:1d} \033[1;34m│".format(expectedoutputs[predictionnumber][0]))
    print("\033[1;34m└───┴───┴─────────┴───┘")


# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

