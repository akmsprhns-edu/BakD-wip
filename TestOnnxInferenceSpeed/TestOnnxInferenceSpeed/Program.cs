﻿using Microsoft.ML;
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
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            var rand = new Random();
            Console.WriteLine("Generationg random data");
            var inputDimensions = new int[] { 250, 15, 15, 2 };

            var inputData = new float[inputDimensions[0] * inputDimensions[1] * inputDimensions[2] * inputDimensions[3]];
            for (int i = 0; i < inputData.Length; i++)
            {
                inputData[i] = (float)rand.NextDouble();
            }
            var threads = new List<Thread>();

            Console.WriteLine("Prepearing threads");
            for (int t = 0; t < 8; t++)
            {
                threads.Add(new Thread(() =>
                {
                    var inferenceSession = new InferenceSession(ModelPath, SessionOptions.MakeSessionOptionWithCudaProvider(0));
                    

                    for (int i = 0; i < 1000; i++)
                    {
                        
                        var inputs = new List<NamedOnnxValue>()
                        {
                            NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(inputData, inputDimensions))
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
    }
}
