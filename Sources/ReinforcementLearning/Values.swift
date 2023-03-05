//
//  Values.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-02.
//

import Foundation

/// Computes discounted returns.
///
/// Discounted returns are defined as follows:
/// `Q_t = \sum_{t'=t}^T gamma^{t'-t} * r_{t'} + gamma^{T-t+1} * finalValue`,
/// where `r_t` represents the reward at time step `t` and `gamma` represents the discount factor.
/// For more details refer to "Reinforcement Learning: An Introduction" Second Edition by
/// Richard S. Sutton and Andrew G. Barto.
///
/// The discounted return computation also takes into account the time steps when episodes end
/// (i.e., steps whose kind is `.last`) by making sure to reset the discounted return being carried
/// backwards through time.
///
/// - Parameters:
///   - discountFactor: Reward discount factor (`gamma` in the above example).
///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
///   - rewards: Contains the rewards for each step.
///   - finalValue: Estimated value at the final step. This is used to bootstrap the reward-to-go
///     computation. Defaults to zeros.
///
/// - Returns: Array of discounted return values over time.
@inlinable
public func discountedReturns(
    discountFactor: Float,
    stepKinds: [StepKind],
    rewards: [Float],
    finalValue: Float? = nil
) -> [Float] {
    let T = stepKinds.count
    let Tminus1 = T - 1
    let finalReward = finalValue ?? 0.0
    var discountedReturns: [Float] = []
    for t in 0..<T {
        let futureReturn = T - t < T ? discountedReturns[t - 1] : finalReward
        let discountedFutureReturn = discountFactor * futureReturn
        let isLast = stepKinds[Tminus1 - t] == .last
        let discountedReturn = rewards[Tminus1 - t] + (isLast ? 0 : discountedFutureReturn)
        discountedReturns.append(discountedReturn)
    }
    return discountedReturns.reversed()
}

/// Advantage estimation result, which contains two tensors:
///   - `advantages`: Estimated advantages that are typically used to train actor networks.
///   - `discountedReturns`: Discounted returns that are typically used to train value networks.
public struct AdvantageEstimate {
    public let advantages: [Float]
    public let discountedReturns: () -> [Float]
    
    @inlinable
    public init(advantages: [Float], discountedReturns: @escaping () -> [Float]) {
        self.advantages = advantages
        self.discountedReturns = discountedReturns
    }
}

public protocol AdvantageFunction {
    /// - Parameters:
    ///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
    ///   - rewards: Contains the rewards obtained at each step.
    ///   - values: Contains the value estimates for each step.
    ///   - finalValue: Estimated value at the final step.
    func callAsFunction(
        stepKinds: [StepKind],
        rewards: [Float],
        values: [Float],
        finalValue: Float
    ) -> AdvantageEstimate
}

/// Performs empirical advantage estimation.
///
/// The empirical advantage estimate at step `t` is defined as:
/// `advantage[t] = returns[t] - value[t]`, where the returns are computed using
/// `discountedReturns(discountFactor:stepKinds:rewards:finalValue:)`.
public struct EmpiricalAdvantageEstimation: AdvantageFunction {
    public let discountFactor: Float
    
    /// - Parameters:
    ///   - discountFactor: Reward discount factor value, which must be between `0.0` and `1.0`.
    @inlinable
    public init(discountFactor: Float) {
        self.discountFactor = discountFactor
    }
    
    /// - Parameters:
    ///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
    ///   - rewards: Contains the rewards obtained at each step.
    ///   - values: Contains the value estimates for each step.
    ///   - finalValue: Estimated value at the final step.
    @inlinable
    public func callAsFunction(
        stepKinds: [StepKind],
        rewards: [Float],
        values: [Float],
        finalValue: Float
    ) -> AdvantageEstimate {
        let returns = discountedReturns(
            discountFactor: discountFactor,
            stepKinds: stepKinds,
            rewards: rewards,
            finalValue: finalValue)
        let advantages = Array(zip(returns, values)).map { $0.0 - $0.1 }
        return AdvantageEstimate(
            advantages: advantages,
            discountedReturns: { () in returns }
        )
    }
}

///// Performs generalized advantage estimation.
/////
///// For more details refer to "High-Dimensional Continuous Control Using Generalized Advantage
///// Estimation" by John Schulman, Philipp Moritz et al. The full paper can be found at:
///// https://arxiv.org/abs/1506.02438.
//public struct GeneralizedAdvantageEstimation: AdvantageFunction {
//    public let discountFactor: Float
//    public let discountWeight: Float
//
//    /// - Parameters:
//    ///   - discountFactor: Reward discount factor value, which must be between `0.0` and `1.0`.
//    ///   - discountWeight: A weight between `0.0` and `1.0` that is used for variance reduction in
//    ///     the temporal differences.
//    @inlinable
//    public init(discountFactor: Float, discountWeight: Float = 1) {
//        self.discountFactor = discountFactor
//        self.discountWeight = discountWeight
//    }
//
//    /// - Parameters:
//    ///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
//    ///   - rewards: Contains the rewards obtained at each step.
//    ///   - values: Contains the value estimates for each step.
//    ///   - finalValue: Estimated value at the final step.
//    @inlinable
//    public func callAsFunction(
//        stepKinds: StepKind,
//        rewards: Float,
//        values: Float,
//        finalValue: Float
//    ) -> AdvantageEstimate {
//        let discountWeight = self.discountWeight
//        let discountFactor = self.discountFactor
//        let isNotLast = 1 - (stepKinds == .last ? 1 : 0)
//        let T = stepKinds.rawValue.shape[0]
//
//        // Compute advantages in reverse order.
//        // TODO: This looks very ugly.
//        let Tminus1 = Tensor(Int64(T - 1))
//        let last = rewards.gathering(atIndices: Tminus1) +
//        discountFactor * finalValue * isNotLast.gathering(atIndices: Tminus1) -
//        values.gathering(atIndices: Tminus1)
//        var advantages = [last]
//        for t in 1..<T {
//            let tTensor = Tensor(Int64(t))
//            let nextValue = values.gathering(atIndices: Tminus1 - tTensor + 1) *
//            isNotLast.gathering(atIndices: Tminus1 - tTensor)
//            let delta = rewards.gathering(atIndices: Tminus1 - tTensor) +
//            discountFactor * nextValue -
//            values.gathering(atIndices: Tminus1 - tTensor)
//            let nextAdvantage = advantages[t - 1] * isNotLast.gathering(atIndices: Tminus1 - tTensor)
//            advantages.append(delta + discountWeight * discountFactor * nextAdvantage)
//        }
//
//        return AdvantageEstimate(
//            advantages: Tensor(advantages.reversed()),
//            discountedReturns: { () in
//                discountedReturns(
//                    discountFactor: discountFactor,
//                    stepKinds: stepKinds,
//                    rewards: rewards,
//                    finalValue: finalValue)
//            })
//    }
//}
