using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlowProject.Criterions;

namespace TensorFlowProject.Optimizers
{
    public class Adam : Optimizer
    {
        public Adam(IVariableV1[] parameters, float learn_rate) : base(parameters)
        {
            Base = keras.optimizers.Adam(learn_rate);
        }
    }
}
