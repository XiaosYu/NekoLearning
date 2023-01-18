using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject.Criterions
{
    public class MSELoss : Criterion
    {
        public override Loss Forward(Tensor @out, Tensor target)
        {
            var loss = tf.reduce_mean(tf.pow(@out - target, 2));
            return new Loss() { Value= loss };
        }
    }
}
