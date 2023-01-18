using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject.Modules
{
    public class ReLU : Module
    {
        public override Tensor Forward(Tensor data)
        {
            return keras.activations.Relu(data);
        }
    }
}
