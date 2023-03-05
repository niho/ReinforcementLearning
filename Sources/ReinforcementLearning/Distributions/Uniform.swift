//
//  Uniform.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public struct Uniform: Distribution {
    public var lowerBound: Float
    public var upperBound: Float
    
    @inlinable
    public init(
        lowerBound: Float = 0.0,
        upperBound: Float = 1.0
    ) {
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }
    
    @inlinable
    public func logProbability(of value: Float) -> Float {
        logf(1.0) - logf(upperBound - lowerBound)
    }
    
    @inlinable
    public func entropy() -> Float {
        logf(upperBound - lowerBound)
    }
    
    @inlinable
    public func mode() -> Float {
        sample()
    }
    
    @inlinable
    public func sample() -> Float {
        return Float.random(in: lowerBound...upperBound)
    }
}
