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
    public class Linear : Module
    {
        public IVariableV1 _parameter { get; }
        public IVariableV1 _bias { get; }

        public Linear(int in_features, int out_features)
        {
            var random_normal = tf.initializers.random_normal_initializer();
            _parameter = tf.Variable(random_normal.Apply(new InitializerArgs((in_features, out_features))));
            _bias = tf.Variable(random_normal.Apply(new InitializerArgs((out_features))));
        }

        public override Tensor Forward(Tensor data)
        {
            var @out = tf.add(tf.matmul(data, _parameter.AsTensor()), _bias);
            return @out;
        }

    }
}
