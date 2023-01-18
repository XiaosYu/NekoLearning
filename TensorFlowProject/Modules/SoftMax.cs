using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject.Modules
{
    public class SoftMax:Module
    {
        public SoftMax(int axis=1)
        {
            this.axis = axis;
        }
        private int axis = 1;
        public override Tensor Forward(Tensor data)
        {
            return keras.activations.Softmax(data);
        }
    }
}
