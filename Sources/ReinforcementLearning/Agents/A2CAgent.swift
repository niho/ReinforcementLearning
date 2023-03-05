//
//  A2CAgent.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

public struct A2CAgent<
    Environment: ReinforcementLearning.Environment,
    State,
    Network: ReinforcementLearning.Network
>: PolicyGradientAgent
where
    Environment.Reward == Float,
    Network.Input == AgentInput<Environment.Observation, State>,
    Network.Output == ActorCriticOutput<Environment.ActionSpace.ValueDistribution, State>
{
    public typealias Observation = Environment.Observation
    public typealias Action = ActionDistribution.Value
    public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
    public typealias Reward = Float
    
    public let actionSpace: Environment.ActionSpace
    public var state: State
    public var network: Network
    
    public let advantageFunction: AdvantageFunction
    public let valueEstimationLossWeight: Float
    public let entropyRegularizationWeight: Float
    
//    @usableFromInline internal var advantagesNormalizer: TensorNormalizer<Float>?
    
    @inlinable
    public init(
        for environment: Environment,
        network: Network,
        initialState: State,
        advantageFunction: AdvantageFunction = EmpiricalAdvantageEstimation(discountFactor: 0.9),
            //GeneralizedAdvantageEstimation(discountFactor: 0.9),
        normalizeAdvantages: Bool = true,
        valueEstimationLossWeight: Float = 0.2,
        entropyRegularizationWeight: Float = 0.0
    ) {
        self.actionSpace = environment.actionSpace
        self.state = initialState
        self.network = network
        self.advantageFunction = advantageFunction
//        self.advantagesNormalizer = normalizeAdvantages ?
//            TensorNormalizer(streaming: true, alongAxes: 0, 1) : nil
        self.valueEstimationLossWeight = valueEstimationLossWeight
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
//        let (loss, gradient) = valueWithGradient(at: network) { network -> Tensor<Float> in
//            let input = AgentInput(
//                observation: trajectory.observation,
//                state: trajectory.state
//            )
//            let networkOutput = try await network.prediction(input)
//
//            // Split the trajectory such that the last step is only used to provide the final value
//            // estimate used for advantage estimation.
//            let sequenceLength = networkOutput.value.shape[0] - 1
//            let stepKinds = StepKind(trajectory.stepKind.rawValue[0..<sequenceLength])
//            let values = networkOutput.value[0..<sequenceLength]
//            let finalValue = networkOutput.value[sequenceLength]
//
//            // Estimate the advantages for the provided trajectory.
//            let advantageEstimate = self.advantageFunction(
//                stepKinds: stepKinds,
//                rewards: trajectory.reward[0..<sequenceLength],
//                values: withoutDerivative(at: values),
//                finalValue: withoutDerivative(at: finalValue))
//            var advantages = advantageEstimate.advantages
////            self.advantagesNormalizer?.update(using: advantages)
////            if let normalizer = self.advantagesNormalizer {
////                advantages = normalizer.normalize(advantages)
////            }
//            let returns = advantageEstimate.discountedReturns()
//
//            // Compute the action log probabilities.
//            let actionDistribution = networkOutput.actionDistribution
//            let actionLogProbs = actionDistribution.logProbability(
//                of: trajectory.action
//            )[0..<sequenceLength]
//
//            // The policy gradient loss is defined as the sum, over time steps, of action
//            // log-probabilities multiplied with the normalized advantages.
//            let actionLogProbWeightedReturns = actionLogProbs * advantages
//            let policyGradientLoss = -actionLogProbWeightedReturns.mean()
//
//            // The value estimation loss is defined as the mean squared error between the value
//            // estimates and the discounted returns.
//            let valueMSE = (values - returns).squared().mean()
//            let valueEstimationLoss = self.valueEstimationLossWeight * valueMSE
//
//            // If entropy regularization is being used for the action distribution, then we also
//            // compute the entropy loss term.
//            var entropyLoss = 0.0
//            if self.entropyRegularizationWeight > 0.0 {
//                let entropy = actionDistribution.entropy()[0..<sequenceLength]
//                entropyLoss = entropyLoss - self.entropyRegularizationWeight * entropy.mean()
//            }
//            return policyGradientLoss + valueEstimationLoss + entropyLoss
//        }
//        optimizer.update(&network, along: gradient)
//        return loss.scalarized()
        return 0.0
    }
}

public struct ActorCriticOutput<ActionDistribution, State> {
    public var actionDistribution: ActionDistribution
    public var value: Float
    public var state: State
    
    @inlinable
    public init(actionDistribution: ActionDistribution, value: Float, state: State) {
        self.actionDistribution = actionDistribution
        self.value = value
        self.state = state
    }
}
