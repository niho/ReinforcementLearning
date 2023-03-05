import XCTest
@testable import ReinforcementLearning

final class ReinforcementLearningTests: XCTestCase {
    func testArgmax() throws {
        XCTAssertEqual(
            [1,43,10,17].argmax(),
            1
        )
    }
    
    func testSoftmax() throws {
        XCTAssertEqual(
            softmax([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]).map { round($0 * 1000) / 1000.0 },
            [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
        )
    }
    
    func testSampling() throws {
        let environment = TestEnvironment()
        for _ in (1...1000) {
            let observation = environment.observationSpace.sample()
            XCTAssert(environment.observationSpace.contains(observation))
        }
    }
    
    func testActorNetwork() async throws {
        let network = TestActorNetwork()
        let observation = TestEnvironment.Observation(fitness: 100.0, fatigue: 0.0)
        let input = AgentInput(observation: observation, state: 0)
        
        let loss = try await network.update(input, with: { output in
            return output
        })
        XCTAssertLessThan(loss, 0.1)
        
        let output = try await network.prediction(input)
        XCTAssertEqual(output.state, 0)
        XCTAssertEqual(output.actionDistribution.actionDistribution.probabilities, [0.0,1.0])
        XCTAssertEqual(output.actionDistribution.probability(of: .workout), 0.0)
        XCTAssertEqual(output.actionDistribution.probability(of: .recovery), 0.0)
    }
    
    func testReinforceAgent() async throws {
        let maxEpisodes: Int = 32
        let maxReplayedSequenceLength: Int = 1000
        var environment = TestEnvironment()
        var agent = ReinforceAgent(
            for: environment,
            network: TestActorNetwork(),
            initialState: 0,
            discountFactor: 0.9,
            entropyRegularizationWeight: 0.01
        )
        try await runSimulation(
            agent: &agent,
            environment: &environment,
            maxEpisodes: maxEpisodes,
            maxReplayedSequenceLength: maxReplayedSequenceLength
        )
        
        var validationEnvironment = TestEnvironment()
        try await agent.run(in: &validationEnvironment, maxSteps: 1000, maxEpisodes: 1, callbacks: [{ (environment, trajectory) in
            print(environment)
            print(trajectory)
        }])
        print("Total steps: \(validationEnvironment.totalSteps) | Total reward: \(validationEnvironment.totalReward))")
    }
    
    func testDQNAgent() async throws {
        let maxEpisodes: Int = 32
        let maxReplayedSequenceLength: Int = 1000
        var environment = TestEnvironment()
        var agent = DQNAgent(
            for: environment,
            network: TestQNetwork(),
            initialState: 0,
            trainSequenceLength: 1,
            maxReplayedSequenceLength: maxReplayedSequenceLength,
            epsilonGreedy: 0.2,
            targetUpdateForgetFactor: 0.95,
            targetUpdatePeriod: 5,
            discountFactor: 0.7,
            trainStepsPerIteration: 10
        )
        try await runSimulation(
            agent: &agent,
            environment: &environment,
            maxEpisodes: maxEpisodes,
            maxReplayedSequenceLength: maxReplayedSequenceLength
        )
    }
    
    private func runSimulation<A: Agent>(
        agent: inout A,
        environment: inout A.Environment,
        maxEpisodes: Int,
        maxReplayedSequenceLength: Int
    ) async throws where A.Environment == TestEnvironment {
        for step in 0..<1000 {
            environment.totalSteps = 0
            environment.totalReward = 0.0
            let loss = try await agent.update(
                using: &environment,
                maxSteps: maxReplayedSequenceLength,
                maxEpisodes: maxEpisodes,
                callbacks: []
            )
            try environment.reset()
            if step % 1 == 0 {
                print("Step \(step) | Loss: \(loss) | Total steps: \(environment.totalSteps) | Total cumulative reward: \(environment.totalReward)")
            }
        }
    }
}
