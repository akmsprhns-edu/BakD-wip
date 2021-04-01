using Microsoft.ML;
using System;
using System.Collections.Generic;

namespace TestOnnxInferenceSpeed
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var mlContext = new MLContext();
            var rand = new Random();
            var dummyData = mlContext.Data.LoadFromEnumerable(new InputData[0]);

            int? gpu = null;
            var pipeline = mlContext.Transforms.ApplyOnnxModel("./simple.onnx");
            var transformer = pipeline.Fit(dummyData);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, Prediction>(transformer);

            Console.WriteLine("Generationg random data");
            var inputDataList = new List<InputData>();
            for(int i = 0; i < 100; i++)
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

            Console.WriteLine("Starting inference");
            var start = DateTime.Now;
            for (int i = 0; i < 1000; i++)
            {
                foreach (var item in inputDataList)
                {
                    _ = predictionEngine.Predict(item).Evaluation;
                }
            }

            var duration = DateTime.Now - start;
            Console.WriteLine($"###############DONE in {duration.TotalSeconds} seconds################");

        }
    }
}
