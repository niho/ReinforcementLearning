//
//  Network.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public protocol Network<Input, Output> {
    associatedtype Input
    associatedtype Output
    
    typealias Loss = Float
    
    func prediction(_ input: Input) async throws -> Output
    func update(_ input: Input, with lossFunc: (Output) async throws -> Output) async throws -> Loss
}

//extension Network {
//    public func update(using: Self, forgetFactor: Float) {
//
//    }
//}
