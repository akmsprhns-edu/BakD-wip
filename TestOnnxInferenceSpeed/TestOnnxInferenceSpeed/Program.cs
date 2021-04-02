using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Threading;

namespace TestOnnxInferenceSpeed
{
    class Program
    {
        private static readonly string ModelPath = "./model.1.onnx";
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");






            var rand = new Random();
            Console.WriteLine("Generationg random data");
            var inputDataList = new List<InputData>();
            for(int i = 0; i < 10000; i++)
            {
                var inputData = new InputData()
                {
                    Input = new float[450]
                };

                for(int j = 0; j < inputData.Input.Length; j++)
                {
                    inputData.Input[j] = rand.Next(2);
                }
                inputDataList.Add(inputData);
            }
            var threads = new List<Thread>();

            var mlContext = new MLContext();
            var dummyData = mlContext.Data.LoadFromEnumerable(new InputData[0]);


            Console.WriteLine("Prepearing threads");
            for (int t = 0; t < 16; t++)
            {
                threads.Add(new Thread(() =>
                {
                    
                    int? gpu = null;
                    var pipeline = mlContext.Transforms.ApplyOnnxModel(ModelPath, gpuDeviceId: gpu);
                    var transformer = pipeline.Fit(dummyData);
                    var predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, Prediction>(transformer);
                    //var inferenceSession = new InferenceSession(ModelPath, SessionOptions.MakeSessionOptionWithCudaProvider(0));
                    //var container = new List<NamedOnnxValue>();

                    for (int i = 0; i < 2; i++)
                    {
                        var data = mlContext.Data.LoadFromEnumerable(inputDataList);
                        foreach (var item in inputDataList)
                        {
                            _ = predictionEngine.Predict(item).Evaluation;
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
