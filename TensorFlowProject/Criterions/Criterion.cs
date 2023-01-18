using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;

namespace TensorFlowProject.Criterions
{
    public abstract class Criterion
    {
        public abstract Loss Forward(Tensor @out, Tensor target);
    }
}
