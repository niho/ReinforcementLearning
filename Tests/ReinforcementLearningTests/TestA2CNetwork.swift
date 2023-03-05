//
//  TestA2CNetwork.swift
//  
//
//  Created by Niklas Holmgren on 2023-03-04.
//

import Foundation
import ReinforcementLearning
import MLCompute

struct TestA2CNetwork: Network {
    private var device: MLCDevice
    private var graph: MLCGraph
    private var inferenceGraph: MLCInferenceGraph
    private var trainingGraph: MLCTrainingGraph
    
    private let batchSize: Int = 1
    private let numberOfActions: Int = TestEnvironment.Action.allCases.count
    
    private let dense1LayerOutputSize: Int = 100
    private let dense2LayerOutputSize: Int = TestEnvironment.Action.allCases.count
    
    public init() {
        self.device = MLCDevice()
        
        let inputTensor1 = MLCTensor(
            descriptor: MLCTensorDescriptor(
                shape: [batchSize, 1],
                dataType: .float32
            )!,
            randomInitializerType: .glorotUniform
        )
        let inputTensor2 = MLCTensor(
            descriptor: MLCTensorDescriptor(
                shape: [batchSize, 1],
                dataType: .float32
            )!,
            randomInitializerType: .glorotUniform
        )
        let lossLabelTensor = MLCTensor(
            descriptor: MLCTensorDescriptor(
                shape: [batchSize, numberOfActions],
                dataType: .float32
            )!
        )
        
        self.graph = MLCGraph()
        let concatTensor = self.graph.node(
            with: MLCConcatenationLayer(),
            sources: [inputTensor1, inputTensor2]
        )
        let dense1 = graph.node(
            with: MLCFullyConnectedLayer(
                weights: MLCTensor(
                    descriptor: MLCTensorDescriptor(
                        shape: [1, 2*dense1LayerOutputSize],
                        dataType: .float32
                    )!,
                    randomInitializerType: .glorotUniform
                ),
                biases: MLCTensor(
                    descriptor: MLCTensorDescriptor(
                        shape: [1, dense1LayerOutputSize],
                        dataType: .float32
                    )!,
                    randomInitializerType: .glorotUniform
                ),
                descriptor: MLCConvolutionDescriptor(
                    kernelSizes: (height: 2, width: dense1LayerOutputSize),
                    inputFeatureChannelCount: 2,
                    outputFeatureChannelCount: dense1LayerOutputSize
                )
            )!,
            sources: [concatTensor!]
        )
        let relu1 = graph.node(
            with: MLCActivationLayer(
                descriptor: MLCActivationDescriptor(
                    type: MLCActivationType.relu
                )!
            ),
            source: dense1!
        )
        let dense2 = graph.node(
            with: MLCFullyConnectedLayer(
                weights: MLCTensor(
                    descriptor: MLCTensorDescriptor(
                        shape: [1, dense1LayerOutputSize*dense2LayerOutputSize],
                        dataType: .float32
                    )!,
                    randomInitializerType: .glorotUniform
                ),
                biases: MLCTensor(
                    descriptor: MLCTensorDescriptor(
                        shape: [1, dense2LayerOutputSize],
                        dataType: .float32
                    )!,
                    randomInitializerType: .glorotUniform
                ),
                descriptor: MLCConvolutionDescriptor(
                    kernelSizes: (height: dense1LayerOutputSize, width: dense2LayerOutputSize),
                    inputFeatureChannelCount: dense1LayerOutputSize,
                    outputFeatureChannelCount: dense2LayerOutputSize
                )
            )!,
            sources: [relu1!]
        )
        let _ = self.graph.node(
            with: MLCSoftmaxLayer(operation: .softmax),
            source: dense2!
        )
        
        self.trainingGraph = MLCTrainingGraph(
            graphObjects: [self.graph],
            lossLayer: MLCLossLayer(
                descriptor: MLCLossDescriptor(
                    type: .softmaxCrossEntropy,
                    reductionType: .mean
                )
            ),
            optimizer: MLCAdamOptimizer(
                descriptor: MLCOptimizerDescriptor(
                    learningRate: 0.001,
                    gradientRescale: 1.0,
                    regularizationType: .none,
                    regularizationScale: 0.0
                ),
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-7,
                timeStep: 1
            )
        )
        self.trainingGraph.addInputs(
            [
                "fitness" : inputTensor1,
                "fatigue" : inputTensor2
            ],
            lossLabels: ["label" : lossLabelTensor]
        )
        self.trainingGraph.compile(options: [], device: self.device)
        
        self.inferenceGraph = MLCInferenceGraph(graphObjects: [self.graph])
        self.inferenceGraph.addInputs([
            "fitness" : inputTensor1,
            "fatigue" : inputTensor2
        ])
        self.inferenceGraph.compile(options: .debugLayers, device: self.device)
    }
    
    func prediction(_ input: AgentInput<TestEnvironment.Observation, Int>) async throws -> ActorCriticOutput<TestEnvironment.ActionDistribution, Int> {
        
        let buffer1: [Float] = [input.observation.fitness]
        let buffer2: [Float] = [input.observation.fatigue]
        
        let data1 = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(buffer1), length: buffer1.count * MemoryLayout<Float>.size)
        let data2 = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(buffer2), length: buffer2.count * MemoryLayout<Float>.size)
        
        return try await withCheckedThrowingContinuation { continuation in
            self.inferenceGraph.execute(
                inputsData: ["fitness" : data1, "fatigue" : data2],
                batchSize: 0,
                options: []
            ) { (resultTensor, error, time) in
                if let error = error {
                    continuation.resume(with: .failure(error))
                    return
                }
                
                let buffer3 = UnsafeMutableRawPointer.allocate(byteCount: self.dense2LayerOutputSize * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)

                resultTensor!.copyDataFromDeviceMemory(toBytes: buffer3, length: self.dense2LayerOutputSize * MemoryLayout<Float>.size, synchronizeWithDevice: false)

                let float4Ptr = buffer3.bindMemory(to: Float.self, capacity: self.dense2LayerOutputSize)
                let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: self.dense2LayerOutputSize)
                
                let actionDistribution = TestEnvironment.ActionDistribution(
                    probabilities: Array(float4Buffer)
                )
                
                let output = ActorCriticOutput(
                    actionDistribution: actionDistribution,
                    value: 0.0,
                    state: input.state
                )
                
                continuation.resume(with: .success(output))
            }
        }
    }
    
    func update(
        _ input: AgentInput<TestEnvironment.Observation, Int>,
        with lossFunc: (ActorCriticOutput<TestEnvironment.ActionDistribution, Int>) async throws -> ActorCriticOutput<TestEnvironment.ActionDistribution, Int>
    ) async throws -> Loss {
        
        let output = try await self.prediction(input)
        let loss = try await lossFunc(output)
        
        let buffer1: [Float] = [input.observation.fitness]
        let buffer2: [Float] = [input.observation.fatigue]
        let buffer3: [[Float]] = [output.actionDistribution.actionDistribution.probabilities]
        
        let data1 = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(buffer1), length: buffer1.count * MemoryLayout<Float>.size)
        let data2 = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(buffer2), length: buffer2.count * MemoryLayout<Float>.size)
        let data3 = MLCTensorData(immutableBytesNoCopy: UnsafeRawPointer(buffer3), length: numberOfActions * buffer3.count * MemoryLayout<Float>.size)
        
        return try await withCheckedThrowingContinuation { continuation in
            self.trainingGraph.execute(
                inputsData: ["fitness": data1, "fatigue": data2],
                lossLabelsData: ["label": data3],
                lossLabelWeightsData: nil,
                batchSize: self.batchSize,
                options: [.synchronous]) { (resultTensor, error, time) in
                    if let error = error {
                        continuation.resume(with: .failure(error))
                    } else {
                        if let data = resultTensor?.data {
                            let float4Array = data.withUnsafeBytes {
                                Array(UnsafeBufferPointer<Float32>(start: $0, count: data.count/MemoryLayout<Float32>.stride))
                            }
                            continuation.resume(with: .success(float4Array[0]))
                        } else {
                            continuation.resume(with: .success(0.0))
                        }
                    }
                }
        }
    }
}
