//
//  DQNAgent.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public struct DQNAgent<
    Environment: ReinforcementLearning.Environment,
    State,
    Network: ReinforcementLearning.Network
>: ProbabilisticAgent
where
    Environment.ActionSpace.ValueDistribution: Discrete,
    Environment.Reward == Float,
    Network.Input == AgentInput<Environment.Observation, State>,
    Network.Output == QNetworkOutput<State>
{
    public typealias Observation = Environment.Observation
    public typealias Action = Environment.ActionSpace.Value
    public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
    public typealias Reward = Float
    
    public var actionSpace: Environment.ActionSpace
    public var state: State
    public var network: Network
    
    public let trainSequenceLength: Int
    public let maxReplayedSequenceLength: Int
    public let epsilonGreedy: Float
    public let targetUpdateForgetFactor: Float
    public let targetUpdatePeriod: Int
    public let discountFactor: Float
    public let trainStepsPerIteration: Int
    
    private var replayBuffer: [Trajectory<Observation, State, Action, Reward>] = []
    
    public init(
        for environment: Environment,
        network: Network,
        initialState: State,
        trainSequenceLength: Int,
        maxReplayedSequenceLength: Int,
        epsilonGreedy: Float = 0.1,
        targetUpdateForgetFactor: Float = 1.0,
        targetUpdatePeriod: Int = 1,
        discountFactor: Float = 0.99,
        trainStepsPerIteration: Int = 1
    ) {
        precondition(
          trainSequenceLength > 0,
          "The provided training sequence length must be greater than 0.")
        precondition(
          trainSequenceLength < maxReplayedSequenceLength,
          "The provided training sequence length is larger than the maximum replayed sequence length.")
        precondition(
          targetUpdateForgetFactor > 0.0 && targetUpdateForgetFactor <= 1.0,
          "The target update forget factor must be in the interval (0, 1].")
        self.actionSpace = environment.actionSpace
        self.state = initialState
        self.network = network
//        self.targetNetwork = network.copy()
        self.trainSequenceLength = trainSequenceLength
        self.maxReplayedSequenceLength = maxReplayedSequenceLength
        self.epsilonGreedy = epsilonGreedy
        self.targetUpdateForgetFactor = targetUpdateForgetFactor
        self.targetUpdatePeriod = targetUpdatePeriod
        self.discountFactor = discountFactor
        self.trainStepsPerIteration = trainStepsPerIteration
    }
    
    public mutating func update(
        using trajectory: Trajectory<Observation, State, Action, Reward>
    ) async throws -> Float {
        guard let step = trajectory.currentStep else { return 0.0 }
        let input = AgentInput(
            observation: step.observation,
            state: step.state
        )
        let loss: Float = try await network.update(input) { output in
//            let targetQValues = output.qValues.map { qValue in
//            if let qValue = output.qValues.max() {
            let nextQValues = try await self.computeNextQValues(
                stepKind: step.stepKind,
                observation: step.observation,
                state: step.state
            )
            let targetQValues = nextQValues.map { nextQValue in
                let currentReward = step.reward
                return currentReward + self.discountFactor * nextQValue
            }
                
                
//                let error = abs(qValue - targetQValue)
//                let delta: Float = 1.0
//                let quadratic = min(error, delta)
//
//                let tdLoss = 0.5 * quadratic * quadratic + delta * (error - quadratic)
//
//                print(tdLoss)
//                return tdLoss
//            }
//            return 0.0
//            }
            
            return QNetworkOutput(
                qValues: targetQValues,
                state: output.state
            )
        }
        return loss
    }
    
    public mutating func update(
        using environment: inout Environment,
        maxSteps: Int,
        maxEpisodes: Int,
        callbacks: [StepCallback<Environment, State>]
    ) async throws -> Float {
        var currentStep = environment.currentStep
        var numSteps = 0
        var numEpisodes = 0
        while numSteps < maxSteps && numEpisodes < maxEpisodes {
            let state = self.state
            let action = try await self.action(for: currentStep, mode: .epsilonGreedy(epsilonGreedy))
            let nextStep = try environment.step(taking: action)
            var trajectory = Trajectory(
                stepKind: nextStep.kind,
                observation: currentStep.observation,
                state: state,
                action: action,
                reward: nextStep.reward)
            if numSteps <= maxReplayedSequenceLength {
                self.replayBuffer.append(trajectory)
            }
            callbacks.forEach { $0(&environment, &trajectory) }
            numSteps += 1
            numEpisodes += nextStep.kind == .last ? 1 : 0
            currentStep = nextStep
        }
        var averageLoss: [Float] = []
        for _ in 0..<trainStepsPerIteration {
            if let trajectory = self.replayBuffer.randomElement() {
                let loss = try await update(using: trajectory)
                averageLoss.append(loss)
            }
        }
        return averageLoss.reduce(0.0, +) / Float(averageLoss.count)
    }
    
    @inlinable
    public mutating func actionDistribution(
      for step: Step<Observation, Reward>
    ) async throws -> ActionDistribution {
        let input = AgentInput(observation: step.observation, state: state)
        let qNetworkOutput = try await network.prediction(input)
        state = qNetworkOutput.state
        return Environment.ActionSpace.ValueDistribution(logits: qNetworkOutput.qValues)
    }
    
    @inlinable
    internal func computeNextQValues(
        stepKind: StepKind,
        observation: Observation,
        state: State
    ) async throws -> [Float] {
        let input = AgentInput(
            observation: observation,
            state: state
        )
        let output = try await network.prediction(input)
        return output.qValues
    }
}

public struct QNetworkOutput<State> {
    public var qValues: [Float]
    public var state: State
    
    @inlinable
    public init(qValues: [Float], state: State) {
        self.qValues = qValues
        self.state = state
    }
}
