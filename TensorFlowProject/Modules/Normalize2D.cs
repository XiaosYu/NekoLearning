using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject.Modules
{
    public class Normalize2D:Module
    {
        public override Tensor Forward(Tensor data)
        {
            return keras.layers.BatchNormalization().Apply(data);
        }
    }
}
