using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace TestOnnxInferenceSpeed
{
    class Program
    {
        private static readonly string ModelPath = "./onnx_inference_model.onnx";
        private const int ITERATIONS = 1000;
        static void Main(string[] args)
        {
            Run();

        }
        static void Run()
        {
            Console.WriteLine("Hello World!");

            var rand = new Random();
            Console.WriteLine("Generationg random data");
            var inputDimensions = new int[] { 250, 15, 15, 2 };
            var inputDataLen = inputDimensions[0] * inputDimensions[1] * inputDimensions[2] * inputDimensions[3];
            var inputData_Array = new float[ITERATIONS][];

            for (int j = 0; j < ITERATIONS; j++)
            {
                var inputData = new float[inputDataLen];
                for (int i = 0; i < inputData.Length; i++)
                {
                    inputData[i] = (float)rand.NextDouble();
                }
                inputData_Array[j] = inputData;
            }
            var threads = new List<Thread>();

            Console.WriteLine("Prepearing threads");
            for (int t = 0; t < 16; t++)
            {
                threads.Add(new Thread(() =>
                {
                    var inferenceSession = new InferenceSession(ModelPath, SessionOptions.MakeSessionOptionWithCudaProvider(0));


                    for (int i = 0; i < ITERATIONS; i++)
                    {

                        var inputs = new List<NamedOnnxValue>()
                        {
                            NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(inputData_Array[i], inputDimensions))
                        };
                        var outputs = new List<string>()
                        {
                            "output"
                        };

                        using (var results = inferenceSession.Run(inputs, outputs))
                        {
                            // manipulate the results
                            var result = results.First();
                            //foreach(var item in result.AsEnumerable<float>())
                            //{
                            //    Console.WriteLine(item);
                            //}
                            var count = result.AsEnumerable<float>().Count();
                        }
                    }
                }));
            }
            Console.WriteLine("Starting inference");
            var start = DateTime.Now;
            foreach (var thread in threads)
                thread.Start();

            foreach (var thread in threads)
                thread.Join();

            var duration = DateTime.Now - start;
            Console.WriteLine($"###############DONE in {duration.TotalSeconds} seconds################");
        }
        
        static void RunWithBuffer()
        {
            Console.WriteLine("Hello World!");

            var rand = new Random();
            Console.WriteLine("Generationg random data");
            var inputDimensions = new int[] { 1000, 15, 15, 2 };
            var outputDimensions = new int[] { inputDimensions[0], 1 };
            var inputDataLen = inputDimensions[0] * inputDimensions[1] * inputDimensions[2] * inputDimensions[3];
            var outputDataLen = outputDimensions[0] * outputDimensions[1];
            var inputData_Array = new float[ITERATIONS][];

            for (int j = 0; j < ITERATIONS; j++)
            {
                var inputData = new float[inputDataLen];
                for (int i = 0; i < inputData.Length; i++)
                {
                    inputData[i] = (float)rand.NextDouble();
                }
                inputData_Array[j] = inputData;
            }
            var threads = new List<Thread>();

            Console.WriteLine("Prepearing threads");
            for (int t = 0; t < 1; t++)
            {
                threads.Add(new Thread(() =>
                {
                    var inferenceSession = new InferenceSession(ModelPath, SessionOptions.MakeSessionOptionWithCudaProvider(0));
                    var bufferInput = new float[inputDataLen];
                    var bufferOutput = new float[outputDataLen];
                    var tensorInput = new DenseTensor<float>(bufferInput, inputDimensions);
                    var tensorOutput = new DenseTensor<float>(bufferOutput, outputDimensions);
                    using (FixedBufferOnnxValue valueInput = FixedBufferOnnxValue.CreateFromTensor(tensorInput),
                        valueOutput = FixedBufferOnnxValue.CreateFromTensor(tensorOutput))
                    {
                        var inputNames = new[] { "input" };
                        var outputNames = new[] { "output" };
                        var inputValues = new[] { valueInput };
                        var outputValues = new[] { valueOutput };

                        for (int i = 0; i < ITERATIONS; i++)
                        {
                            inputData_Array[i].CopyTo(bufferInput, 0);

                            var outputs = new List<string>()
                                {
                                    "output"
                                };

                            inferenceSession.Run(inputNames, inputValues, outputNames, outputValues);
                        }
                    }
                }));
            }
            Console.WriteLine("Starting inference");
            var start = DateTime.Now;
            foreach (var thread in threads)
                thread.Start();

            foreach (var thread in threads)
                thread.Join();

            var duration = DateTime.Now - start;
            Console.WriteLine($"###############DONE in {duration.TotalSeconds} seconds################");
        }
    }
}
