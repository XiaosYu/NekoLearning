using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject
{
    public class DNN:Module
    {
        public DNN() 
        {
            In_Layer = new Sequential
            (
                new Linear(2, 16),
                new Linear(16, 32),
                new Dropout(),
                new ReLU()
            );

            Hidden_Layer = new Sequential
            (
                new ResBlock
                (
                    new Sequential
                    (
                        new Linear(32, 32),
                        new Linear(32, 32)
                    ),
                    new ReLU()
                ),
                new ResBlock
                (
                    new Sequential
                    (
                        new Linear(32, 32),
                        new Linear(32, 32)
                    ),
                    new ReLU()
                ),
                new ResBlock
                (
                    new Sequential
                    (
                        new Linear(32, 32),
                        new Linear(32, 32)
                    ),
                    new ReLU()
                )
            ); ;

            Out_Layer = new Sequential
            (
                new Dropout(),
                new Linear(32, 16),
                new Linear(16, 2),
                new SoftMax()
            );
        }

        public Module In_Layer { get; set; }
        public Module Hidden_Layer { get; set; }
        public Module Out_Layer { get; set; }

        public override Tensor Forward(Tensor data)
        {
            var @out = this.In_Layer.Forward(data);
            @out = this.Hidden_Layer.Forward(@out);
            @out = this.Out_Layer.Forward(@out);
            return @out;
        }
    }
}
