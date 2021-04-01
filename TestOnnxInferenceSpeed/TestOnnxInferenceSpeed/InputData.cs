using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestOnnxInferenceSpeed
{
    public class InputData
    {
        [ColumnName("input")]
        [VectorType(450)]
        public float[] Input { get; set; }

    }
}
