//
//  DeepNeuralNetwork.swift
//  DeepNeuralNetworkDemo
//
//  Created by Jeremi Kaczmarczyk on 27/09/2017.
//  Copyright © 2017 Jeremi Kaczmarczyk. All rights reserved.
//

import Foundation
import Matswift


class DeepNeuralNetwork {
    
    let iterations: Int
    let learningRate: Double
    
    private var layers = [Layer]()
    
    init(iterations: Int, learningRate: Double) {
        self.iterations = iterations
        self.learningRate = learningRate
    }
    
    func add(layer: Layer) {
        layers.append(layer)
    }
    
    func compile() {
        for i in 0..<layers.count {
            layers[i].initialize(
                previous: i == 0 ? nil : layers[i - 1],
                next: i == layers.count - 1 ? nil : layers[i + 1]
            )
        }
    }
    
    private func layersForward(X: Matrix) -> Matrix {
        var output = X
        for layer in layers {
            output = layer.forward(X: output)
        }
        return output
    }
    
    private func layersBackward(y: Matrix) {
        let m = Double(y.shape.columns)
        for layer in layers.reversed() {
            layer.backward(m: m, y: layer === layers.last ? y : nil)
        }
    }
    
    private func layersUpdate() {
        for layer in layers {
            layer.update(learningRate: learningRate)
        }
    }
    
    private func cost(yHat: Matrix, y: Matrix) -> Double {
        let logprobs = yHat.log() * y
        let cost = -logprobs.sum()
        return cost
    }
    
    func fit(X: Matrix, y: Matrix) {
        for i in 1...iterations {
            let AL = layersForward(X: X)
            let c = cost(yHat: AL, y: y)
            layersBackward(y: y)
            layersUpdate()
            
            if i % 100 == 0 {
                print("Iteration: \(i)")
                print("Cost: \(c)")
                var goodClassification = 0
                for (yHat, y) in zip(AL.values, y.values) {
                    let yHatt = yHat > 0.5 ? 1.0 : 0.0
                    if y == yHatt {
                        goodClassification += 1
                    }
                }
                print("Accuracy: \(Double(goodClassification) / Double(y.values.count))")
            }
        }
    }
    
    func test(X: Matrix, y: Matrix) {
        print("Testing")
        let yHat = layersForward(X: X)
        var goodClassification = 0
        for (yHat, y) in zip(yHat.values, y.values) {
            let prediction = yHat > 0.5 ? 1.0 : 0.0
            print("Prediction: \(prediction), Real: \(y)")
            if y == prediction {
                goodClassification += 1
            }
        }
        print("Accuracy: \(Double(goodClassification) / Double(y.values.count))")
    }
}
