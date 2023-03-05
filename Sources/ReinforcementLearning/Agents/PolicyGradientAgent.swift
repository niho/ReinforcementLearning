//
//  PolicyGradientAgent.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public protocol PolicyGradientAgent: ProbabilisticAgent {}

extension PolicyGradientAgent {
    @inlinable
    @discardableResult
    public mutating func update(
        using environment: inout Environment,
        maxSteps: Int = Int.max,
        maxEpisodes: Int = Int.max,
        callbacks: [StepCallback<Environment, State>] = []
    ) async throws -> Float {
        var trajectory = Trajectory<Observation, State, Action, Reward>()
        var currentStep = environment.currentStep
        var numSteps = 0
        var numEpisodes = 0
        while numSteps < maxSteps && numEpisodes < maxEpisodes {
            let state = self.state
            let action = try await self.action(for: currentStep, mode: .probabilistic)
            let nextStep = try environment.step(taking: action)
            trajectory.append(
                stepKind: nextStep.kind,
                observation: currentStep.observation,
                state: state,
                action: action,
                reward: nextStep.reward
            )
            callbacks.forEach { $0(&environment, &trajectory) }
            numSteps += 1
            numEpisodes += nextStep.kind == .last ? 1 : 0
            currentStep = nextStep
        }
        return try await update(using: trajectory)
    }
}
