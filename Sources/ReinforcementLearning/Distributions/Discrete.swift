//
//  Discrete.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-03.
//

import Foundation

public protocol Discrete: Distribution {
    init(probabilities: [Float])
    init(logits: [Float])
}
