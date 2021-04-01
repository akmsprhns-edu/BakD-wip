using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestOnnxInferenceSpeed
{
    public class Prediction
    {
        [VectorType(1)]
        [ColumnName("output")]
        public float[] Output { get; set; }

        public float Evaluation { get => Output[0]; }
    }
}
