using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject.Modules
{
    public class Tanh : Module
    {
        public override Tensor Forward(Tensor data)
        {
            return (tf.exp(data) - tf.exp(-data))/(tf.exp(data) + tf.exp(data));
        }
    }
}
