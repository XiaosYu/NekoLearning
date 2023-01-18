using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using TensorFlowProject.Criterions;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowProject.Losses
{
    public class CrossEntropyLoss : Criterion
    {
        public CrossEntropyLoss()
        {

        }

        public override Loss Forward(Tensor @out, Tensor target)
        {
            target = tf.one_hot(target, depth: (int)@out.dims[1]);
            @out = tf.clip_by_value(@out, 1e-9f, 1.0f);
            var loss = tf.reduce_mean(-tf.reduce_sum(target * tf.math.log(@out)));
            return new Loss() { Value = loss };
        }
    }
}
