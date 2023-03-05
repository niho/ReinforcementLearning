//
//  Distribution.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public protocol Distribution {
    associatedtype Value
    
    func logProbability(of value: Value) -> Float
    func entropy() -> Float
    
    /// Returns the mode of this distribution. If the distribution has multiple modes, then one of
    /// them is sampled randomly (and uniformly) and returned.
    func mode() -> Value
    
    /// Returns a random sample drawn from this distribution.
    func sample() -> Value
}

public extension Distribution {
    func probability(of value: Value) -> Float {
        exp(logProbability(of: value))
    }
}
