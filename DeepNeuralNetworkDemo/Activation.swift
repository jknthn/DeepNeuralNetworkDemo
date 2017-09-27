//
//  Activation.swift
//  DeepNeuralNetworkDemo
//
//  Created by Jeremi Kaczmarczyk on 27/09/2017.
//  Copyright © 2017 Jeremi Kaczmarczyk. All rights reserved.
//

import Foundation
import Matswift

enum Activation {
    case sigmoid
    case relu
    case none
    
    var forward: (Matrix) -> Matrix {
        switch self {
        case .sigmoid:
            return { matrix in
                let ex = Matrix(values: matrix.values.map { exp($0) }, shape: matrix.shape)
                return ex / (ex + 1.0)
            }
        case .relu:
            return { matrix in
                let newValues = matrix.values.map { max(0.0, $0) }
                return Matrix(values: newValues, shape: matrix.shape)
            }
        case .none:
            return { matrix in
                return matrix
            }
        }
    }
    
    var backward: (Matrix) -> Matrix {
        switch self {
        case .sigmoid:
            return { matrix in
                return self.forward(matrix) * (1 - self.forward(matrix))
            }
        case .relu:
            return { matrix in
                let newValues = matrix.values.map { $0 > 0.0 ? 1.0 : 0.0 }
                return Matrix(values: newValues, shape: matrix.shape)
            }
        case .none:
            return { matrix in
                return matrix
            }
        }
    }
}
