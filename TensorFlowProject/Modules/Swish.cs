using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Operations.Initializers;

namespace TensorFlowProject.Modules
{
    public class Swish:Module
    {
        public Swish()
        {
            var random_normal = tf.initializers.random_normal_initializer();
            beta = tf.Variable(random_normal.Apply(new InitializerArgs((1))));
        }

        public IVariableV1 beta { set; get; }

        public override Tensor Forward(Tensor data)
        {
            return data / (1 + tf.exp(-beta.AsTensor() * data));
        }
    }
}
