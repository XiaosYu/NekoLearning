using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.NumPy;
using Tensorflow.Keras.Optimizers;
using System.Xml.Linq;

namespace TensorFlowProject.Modules
{
    public class Flatten : Module
    {
        public Flatten()
        {

        }

        public override Tensor Forward(Tensor data)
        {
            var @out = tf.reshape(data, new Shape(data.dims[0], -1));
            return @out;
        }
    }
}
