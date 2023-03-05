//
//  Statistics.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public func softmax(_ x: [Float]) -> [Float] {
    let a: [Float] = x.map { expf($0) }
    let b: Float = x.map { expf($0) }.reduce(0.0, +)
    return a.map { $0 / b }
}

public func logSoftmax(_ x: [Float]) -> [Float] {
    return softmax(x).map { logf($0) }
}

extension Array where Element: Comparable {
    func argmax() -> Index? {
        return indices.max(by: { self[$0] < self[$1] })
    }
    
    func argmin() -> Index? {
        return indices.min(by: { self[$0] < self[$1] })
    }
}
