using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowProject
{
    public class FCNN:Module
    {
        public Module In_Layer { get; }
        public Module Hidden_Layer { get; }
        public Module Out_Layer { get; }
        public FCNN()
        {
            In_Layer = new Sequential
            (
                new Flatten(),
                new Linear(28 * 28, 1000),
                new Sigmoid()
            );
            Hidden_Layer = new Sequential
            (
                new Linear(1000, 1000),
                new Sigmoid(),
                new Linear(1000, 1000),
                new Sigmoid(),
                new Linear(1000, 1000),
                new Sigmoid(),
                new Linear(1000, 1000),
                new Sigmoid()
            );
            Out_Layer = new Sequential
            (
                new Linear(1000, 10),
                new SoftMax(1)
            );
        }

        public override Tensor Forward(Tensor data)
        {
            var @out = this.In_Layer.Forward(data);
            @out = this.Hidden_Layer.Forward(@out);
            @out = this.Out_Layer.Forward(@out);
            return @out;
        }

        

    
    }
}
