//
//  File.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation
import ReinforcementLearning

struct TestEnvironment: Environment {
    var observationSpace: ObservationSpace
    var actionSpace: ActionSpace
    
    @usableFromInline internal var step: Step<Observation, Float>
    @usableFromInline internal var totalSteps: Int
    @usableFromInline internal var totalReward: Float
    
    @inlinable var currentStep: Step<Observation, Float> { step }
    
    init() {
        self.observationSpace = ObservationSpace()
        self.actionSpace = ActionSpace()
        self.step = Step(
            kind: .first,
            observation: Observation(fitness: 0, fatigue: 0),
            reward: 0
        )
        self.totalSteps = 0
        self.totalReward = 0.0
    }
    
    @discardableResult
    mutating func step(taking action: Action) throws -> Step<Observation, Float> {
        precondition(actionSpace.contains(action), "Invalid action provided.")
        
        if self.step.kind == .last {
            try self.reset()
        }
        
        let effort: Float = 100.0 //Float.random(in: 90.0...100.0)
        
        if action == .workout {
            step.observation.fitness += effort
            step.observation.fatigue += effort
        } else if action == .recovery {
            step.observation.fatigue -= effort
        }
        
        if step.observation.fatigue >= 300.0 {
            step.observation.fitness = 0.0
            step.observation.fatigue = 0.0
        }
        
        if step.observation.fitness >= 1000.0 {
            step.reward = 1.0
            step.kind = .last
        } else if step.observation.fatigue < -500.0 {
            step.kind = .last
            step.observation.fatigue = 0.0
            step.reward = -1.0
        } else {
            step.reward = -(effort / 1000.0)
            step.kind = .transition
        }
        
        totalSteps += 1
        totalReward += step.reward
        
        return step
    }

    /// Resets the environment.
    @discardableResult
    mutating func reset() throws -> Step<Observation, Float> {
        self.step.kind = StepKind.first
        self.step.observation = Observation(fitness: 0, fatigue: 0)
        self.step.reward = 0
        return step
    }
}

extension TestEnvironment {
    struct Observation {
        var fitness: Float
        var fatigue: Float
    }
    
    struct ObservationSpace: Space {
        let distribution: ValueDistribution
        
        init() {
            self.distribution = ValueDistribution()
        }
        
        func contains(_ value: Observation) -> Bool {
            value.fitness >= 0.0 && value.fatigue >= 0.0
        }
    }
    
    struct ValueDistribution: Distribution {
        var fitnessDistribution = Uniform(lowerBound: 0.0, upperBound: 1000.0)
        var fatigueDistribution = Uniform(lowerBound: -500.0, upperBound: 300.0)
        
        func logProbability(of value: Observation) -> Float {
            fitnessDistribution.logProbability(of: value.fitness) + fatigueDistribution.logProbability(of: value.fatigue)
        }
        
        func entropy() -> Float {
            fitnessDistribution.entropy() + fatigueDistribution.entropy()
        }

        func mode() -> Observation {
            return Observation(
                fitness: fitnessDistribution.mode(),
                fatigue: fatigueDistribution.mode()
            )
        }

        func sample() -> Observation {
            return Observation(
                fitness: fitnessDistribution.sample(),
                fatigue: fatigueDistribution.sample()
            )
        }
    }
    
    enum Action: Int, CaseIterable {
        case workout
        case recovery
    }
    
    struct ActionSpace: Space {
        let distribution: ActionDistribution
        
        init() {
            self.distribution = ActionDistribution()
        }
        
        func contains(_ value: Action) -> Bool {
            true
        }
    }
    
    struct ActionDistribution: Discrete {
        var actionDistribution: Categorical
        
        init() {
            self.actionDistribution = Categorical(probabilities: [0.5, 0.5])
        }
        
        init(probabilities: [Float]) {
            self.actionDistribution = Categorical(probabilities: probabilities)
        }
        
        init(logits: [Float]) {
            self.actionDistribution = Categorical(logits: logits)
        }

        func logProbability(of value: Action) -> Float {
            return actionDistribution.logProbability(of: value.rawValue)
        }

        func entropy() -> Float {
            return actionDistribution.entropy()
        }

        func mode() -> Action {
            return Action(rawValue: actionDistribution.mode())!
        }

        func sample() -> Action {
            return Action(rawValue: actionDistribution.sample())!
        }
    }
}
