//
//  Spaces.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public protocol Space {
    associatedtype Value
    associatedtype ValueDistribution: Distribution where ValueDistribution.Value == Value
    
    var distribution: ValueDistribution { get }
    
    func contains(_ value: Value) -> Bool
}

public extension Space {
    func sample() -> Value {
        distribution.sample()
    }
}
