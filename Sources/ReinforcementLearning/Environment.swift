//
//  Environment.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public protocol Environment {
    associatedtype ObservationSpace: Space
    associatedtype ActionSpace: Space
    associatedtype Reward
    
    typealias Observation = ObservationSpace.Value
    typealias Action = ActionSpace.Value
    
    var observationSpace: ObservationSpace { get }
    var actionSpace: ActionSpace { get }
    
    var currentStep: Step<Observation, Reward> { mutating get }
    
    @discardableResult
    mutating func step(taking action: Action) throws -> Step<Observation, Reward>

    /// Resets the environment.
    @discardableResult
    mutating func reset() throws -> Step<Observation, Reward>
}

public struct Step<Observation, Reward> {
    public var kind: StepKind
    public var observation: Observation
    public var reward: Reward
    
    public init(kind: StepKind, observation: Observation, reward: Reward) {
        self.kind = kind
        self.observation = observation
        self.reward = reward
    }
}

public enum StepKind {
    case first
    case transition
    case last
}
