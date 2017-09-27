//
//  Layer.swift
//  DeepNeuralNetworkDemo
//
//  Created by Jeremi Kaczmarczyk on 27/09/2017.
//  Copyright © 2017 Jeremi Kaczmarczyk. All rights reserved.
//

import Foundation
import Matswift

class Layer {
    
    let size: Int
    let activation: Activation
    
    var weights: Matrix!
    var biases: Matrix!
    
    var previousLayer: Layer?
    var nextLayer: Layer?
    
    var dW: Matrix!
    var db: Matrix!
    var dZ: Matrix!
    
    var A: Matrix!
    var Z: Matrix!
    
    init(size: Int, activation: Activation) {
        self.size = size
        self.activation = activation
    }
    
    func initialize(previous: Layer?, next: Layer?) {
        if let previous = previous {
            weights = Matrix(random: Shape(rows: size, columns: previous.size), multiplier: 0.01)
            biases = Matrix(zeros: Shape(rows: size, columns: 1))
        }
        previousLayer = previous
        nextLayer = next
    }
    
    static func input(size: Int) -> Layer {
        return Layer(size: size, activation: .none)
    }
    
    static func fullyConnected(size: Int, activation: Activation) -> Layer {
        return Layer(size: size, activation: activation)
    }
    
    func forward(X: Matrix) -> Matrix {
        if activation != .none {
            Z = weights.dot(X) + biases
            A = activation.forward(Z)
        } else {
            A = X
        }
        return A
    }
    
    func backward(m: Double, y: Matrix? = nil) {
        if let nextLayer = nextLayer, let previousLayer = previousLayer {
            dZ = nextLayer.weights.T.dot(nextLayer.dZ) * activation.backward(Z)
            dW = (1.0 / m) * (dZ.dot(previousLayer.A.T))
            db = (1.0 / m) * (dZ.sum(direction: .rows))
        } else if let previousLayer = previousLayer, let Y = y {
            dZ = ((Y / A) - ((1 - Y) / (1 - A))).invertSign()
            dW = (1.0 / m) * (dZ.dot(previousLayer.A.T))
            db = (1.0 / m) * (dZ.sum(direction: .rows))
        }
    }
    
    func update(learningRate: Double) {
        if previousLayer != nil {
            weights = weights - (learningRate * dW)
            biases = biases - (learningRate * db)
        }
    }
}
