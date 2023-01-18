using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowProject.Modules
{
    public class Sigmoid : Module
    {
        public Sigmoid()
        {

        }


        public override Tensor Forward(Tensor data)
        {
            return 1 / (1 + tf.exp(-data));
        }

    
    }
}
