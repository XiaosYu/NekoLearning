using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject
{
    public class CNN : Module
    {
        public Module Hidden_Layer { get; }
        public Module Out_Layer { get; }
        public CNN()
        {

            Hidden_Layer = new Sequential
            (
                new Conv2D(3, 1, 2, 1),
                new Normalize2D(),
                new ReLU(),
                new Conv2D(3, 2, 4, 1),
                new Normalize2D(),
                new ReLU(),
                new Conv2D(3, 4, 8, 1),
                new Normalize2D(),
                new ReLU(),
                new Conv2D(3, 8, 16, 1),
                new Normalize2D(),
                new ReLU()
            );
            Out_Layer = new Sequential
            (
                new Flatten(),
                new Linear(6400, 1000),
                new Linear(1000, 10),
                new SoftMax()
            );
        }

        public override Tensor Forward(Tensor data)
        {
            var @out = tf.reshape(data, new Shape(-1, 28, 28, 1));
            @out = this.Hidden_Layer.Forward(@out);
            @out = this.Out_Layer.Forward(@out);
            return @out;
        }




    }
}
