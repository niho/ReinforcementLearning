//
//  ReinforceAgent.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public struct ReinforceAgent<
    Environment: ReinforcementLearning.Environment,
    State,
    Network: ReinforcementLearning.Network
>: PolicyGradientAgent
where
//Environment.ActionSpace.ValueDistribution: DifferentiableDistribution,
Environment.Reward == Float,
Network.Input == AgentInput<Environment.Observation, State>,
Network.Output == ActorOutput<Environment.ActionSpace.ValueDistribution, State>
{
    public typealias Observation = Environment.Observation
    public typealias Action = ActionDistribution.Value
    public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
    public typealias Reward = Float
    
    public let actionSpace: Environment.ActionSpace
    public var state: State
    public var network: Network
    
    public let discountFactor: Float
    public let entropyRegularizationWeight: Float
    
//    @usableFromInline internal var returnsNormalizer: TensorNormalizer<Float>?
    
    @inlinable
    public init(
        for environment: Environment,
        network: Network,
        initialState: State,
        discountFactor: Float,
        normalizeReturns: Bool = true,
        entropyRegularizationWeight: Float = 0.0
    ) {
        self.actionSpace = environment.actionSpace
        self.state = initialState
        self.network = network
        self.discountFactor = discountFactor
//        self.returnsNormalizer = normalizeReturns ?
//        TensorNormalizer(streaming: true, alongAxes: 0, 1) :
//        nil
        self.entropyRegularizationWeight = entropyRegularizationWeight
    }
    
    @inlinable
    public mutating func actionDistribution(
        for step: Step<Observation, Reward>
    ) async throws -> ActionDistribution {
        let input = AgentInput(observation: step.observation, state: state)
        let networkOutput = try await network.prediction(input)
        state = networkOutput.state
        return networkOutput.actionDistribution
    }
    
    @discardableResult
    public mutating func update(
        using trajectory: Trajectory<Observation, State, Action, Reward>
    ) async throws -> Float {
        let returns = discountedReturns(
            discountFactor: discountFactor,
            stepKinds: trajectory.steps.map { $0.stepKind },
            rewards: trajectory.steps.map { $0.reward }
        )
        let cumReturn = returns.reduce(0.0, +)
        guard let step = trajectory.currentStep else { return 0.0 }
        let input = AgentInput(
            observation: step.observation,
            state: step.state
        )
        let loss = try await network.update(input) { output in
            // REINFORCE requires completed episodes and thus we mask out incomplete ones.
            guard trajectory.numEpisodes > 0 else { return output }
                
            let actionDistribution = output.actionDistribution
//            self.returnsNormalizer?.update(using: returns)
//            if let normalizer = self.returnsNormalizer {
//                returns = normalizer.normalize(returns)
//            }
//            let actionLogProb = actionDistribution.logProbability(of: step.action)
//
//            // The policy gradient loss is defined as the sum, over time steps, of action
//            // log-probabilities multiplied with the cumulative return from that time step onward.
//            let actionLogProbWeightedReturn = actionLogProb * cumReturn
//
////            let mask = Tensor<Float>(trajectory.stepKind.completeEpisodeMask())
////            let episodeCount = trajectory.stepKind.episodeCount()
////
////            precondition(
////                episodeCount.scalarized() > 0,
////                "REINFORCE requires at least one completed episode.")
//
//            // We compute the mean of the policy gradient loss over the number of episodes.
//            let policyGradientLoss = -actionLogProbWeightedReturn / Float(trajectory.numEpisodes)
//
//            // If entropy regularization is being used for the action distribution, then we also
//            // compute the entropy loss term.
//            var entropyLoss: Float = 0.0
//            if self.entropyRegularizationWeight > 0.0 {
//                let entropy = actionDistribution.entropy()
//                entropyLoss = entropyLoss - self.entropyRegularizationWeight * entropy
//            }
//            return policyGradientLoss + entropyLoss
            
            return ActorOutput(
                actionDistribution: actionDistribution,
                state: output.state
            )
        }
        return loss
//        let (loss, gradient) = valueWithGradient(at: network) { network -> Tensor<Float> in
//            let networkOutput = network.prediction(AgentInput(
//                observation: trajectory.observation,
//                state: trajectory.state))
//            let actionDistribution = networkOutput.actionDistribution
////            self.returnsNormalizer?.update(using: returns)
////            if let normalizer = self.returnsNormalizer {
////                returns = normalizer.normalize(returns)
////            }
//            let actionLogProbs = actionDistribution.logProbability(of: trajectory.action)
//
//            // The policy gradient loss is defined as the sum, over time steps, of action
//            // log-probabilities multiplied with the cumulative return from that time step onward.
//            let actionLogProbWeightedReturns = actionLogProbs * returns
//
//            // REINFORCE requires completed episodes and thus we mask out incomplete ones.
//            let mask = Tensor<Float>(trajectory.stepKind.completeEpisodeMask())
//            let episodeCount = trajectory.stepKind.episodeCount()
//
//            precondition(
//                episodeCount.scalarized() > 0,
//                "REINFORCE requires at least one completed episode.")
//
//            // We compute the mean of the policy gradient loss over the number of episodes.
//            let policyGradientLoss = -(actionLogProbWeightedReturns * mask).sum() / episodeCount
//
//            // If entropy regularization is being used for the action distribution, then we also
//            // compute the entropy loss term.
//            var entropyLoss = 0.0
//            if self.entropyRegularizationWeight > 0.0 {
//                let entropy = actionDistribution.entropy()
//                entropyLoss = entropyLoss - self.entropyRegularizationWeight * entropy.mean()
//            }
//            return policyGradientLoss + entropyLoss
//        }
//        optimizer.update(&network, along: gradient)
//        return loss.scalarized()
    }
}

public struct ActorOutput<
    ActionDistribution,
    State
> {
    public var actionDistribution: ActionDistribution
    public var state: State
    
    @inlinable
    public init(actionDistribution: ActionDistribution,  state: State) {
        self.actionDistribution = actionDistribution
        self.state = state
    }
}
