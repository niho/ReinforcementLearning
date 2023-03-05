//
//  ProbabilisticAgent.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public enum ProbabilisticAgentMode {
  case random
  case greedy
  case epsilonGreedy(_ epsilon: Float)
  case probabilistic
}

public protocol ProbabilisticAgent: Agent {
  associatedtype ActionDistribution: Distribution where ActionDistribution.Value == Action

  /// Generates the distribution over next actions given the current environment step.
  mutating func actionDistribution(for step: Step<Observation, Reward>) async throws -> ActionDistribution
}

extension ProbabilisticAgent {
  @inlinable
  public mutating func action(for step: Step<Observation, Reward>) async throws -> Action {
    try await action(for: step, mode: .greedy)
  }

  /// - Note: We cannot use a default argument value for `mode` here because of the `Agent`
  ///   protocol requirement for an `Agent.action(for:)` function.
  @inlinable
  public mutating func action(
    for step: Step<Observation, Reward>,
    mode: ProbabilisticAgentMode
  ) async throws -> Action {
    switch mode {
    case .random:
      return actionSpace.sample()
    case .greedy:
      return try await actionDistribution(for: step).mode()
    case let .epsilonGreedy(epsilon) where Float.random(in: 0..<1) < epsilon:
      return actionSpace.sample()
    case .epsilonGreedy(_):
      return try await actionDistribution(for: step).mode()
    case .probabilistic:
      return try await actionDistribution(for: step).sample()
    }
  }

  @inlinable
  public mutating func run(
    in environment: inout Environment,
    mode: ProbabilisticAgentMode = .greedy,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment, State>] = []
  ) async throws {
    var currentStep = environment.currentStep
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let action = try await self.action(for: currentStep, mode: mode)
      let nextStep = try environment.step(taking: action)
      var trajectory = Trajectory(
        stepKind: nextStep.kind,
        observation: currentStep.observation,
        state: state,
        action: action,
        reward: nextStep.reward)
      callbacks.forEach { $0(&environment, &trajectory) }
      numSteps += 1
      numEpisodes += nextStep.kind == .last ? 1 : 0
      currentStep = nextStep
    }
  }
}
