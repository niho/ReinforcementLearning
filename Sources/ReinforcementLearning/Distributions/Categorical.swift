//
//  Categorical.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public struct Categorical: Discrete {
    public var probabilities: [Float]
    
//    public init(logProbabilities: [Float]) {
//        self.logProbabilities = logProbabilities
//    }
    
    public init(probabilities: [Float]) {
        self.probabilities = probabilities
    }
    
    public init(logits: [Float]) {
        self.probabilities = softmax(logits)
    }
    
    public func logProbability(of value: Int) -> Float {
        logf(probabilities[value])
    }
    
    public func entropy() -> Float {
        probabilities.map { -($0 * exp($0)) }.reduce(0.0, +)
    }
    
    public func mode() -> Int {
        probabilities.argmax() ?? 0
    }
    
    public func sample() -> Int {
        let sum = probabilities.reduce(0, +)
        // Random number in the range 0.0 <= rnd < sum :
        let rnd = Float.random(in: 0.0 ..< sum)
        // Find the first interval of accumulated probabilities into which `rnd` falls:
        var accum: Float = 0.0
        for (i, p) in probabilities.enumerated() {
            accum += p
            if rnd < accum {
                return i
            }
        }
        // This point might be reached due to floating point inaccuracies:
        return (probabilities.count - 1)
    }
}
