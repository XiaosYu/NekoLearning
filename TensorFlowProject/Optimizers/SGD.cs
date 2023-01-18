using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Gradients;
using TensorFlowProject.Criterions;
using TensorFlowProject.Losses;

namespace TensorFlowProject.Optimizers
{
    public class SGD : Optimizer
    {
        public SGD(IVariableV1[] parameters, float learn_rate)
            :base(parameters)
        {
            Base = keras.optimizers.SGD(learn_rate);
        }
    }
}
