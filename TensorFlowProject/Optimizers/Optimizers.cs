using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Gradients;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using TensorFlowProject.Criterions;

namespace TensorFlowProject.Optimizers
{
    public abstract class Optimizer
    {
        public Optimizer(IVariableV1[] parameters)
        {
            Parameters = parameters;
        }
        public static IVariableV1[] Parameters { get; private set; }
        protected OptimizerV2 Base { get; set; }
        public static GradientTape Tape { get; set; }
        public static Tensor[] Gradients { get; set; }
        public void ZeroGrad()
        {
            Tape = tf.GradientTape();
        }
        public void Step()
        {
            Base.apply_gradients(zip(Gradients, Parameters.Select(x => x as ResourceVariable)));
        }
        
    }
}
