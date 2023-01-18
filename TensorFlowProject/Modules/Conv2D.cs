using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Operations.Initializers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;


namespace TensorFlowProject.Modules
{
    public class Conv2D:Module
    {
        public Conv2D(int kernel, int in_channel, int out_channel, int stride)
        {
            var random_normal = tf.initializers.random_normal_initializer();
            this.stride = stride;
            this.Filter = tf.Variable(random_normal.Apply(new InitializerArgs((kernel, kernel, in_channel, out_channel))));
        }

        public IVariableV1 Filter { get; }
        private int stride { set; get; }

        public override Tensor Forward(Tensor data)
        {
            var @out = tf.nn.conv2d(data, this.Filter.AsTensor(), new int[] {1, stride, stride, 1 }, "VALID");
            return @out;
        }
    }
}
