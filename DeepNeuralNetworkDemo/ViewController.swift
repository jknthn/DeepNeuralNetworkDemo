//
//  ViewController.swift
//  DeepNeuralNetworkDemo
//
//  Created by Jeremi Kaczmarczyk on 27/09/2017.
//  Copyright © 2017 Jeremi Kaczmarczyk. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        runNeuralNetwork()
    }
    
    func runNeuralNetwork() {
        let data = CSVProcessor()
        
        let nn = DeepNeuralNetwork(iterations: 5000, learningRate: 0.003)
        nn.add(layer: Layer.input(size: 4))
        nn.add(layer: Layer.fullyConnected(size: 8, activation: .relu))
        nn.add(layer: Layer.fullyConnected(size: 1, activation: .sigmoid))
        nn.compile()
        
        nn.fit(X: data.XTrain, y: data.yTrain)
        nn.test(X: data.XTest, y: data.yTest)
    }
}

