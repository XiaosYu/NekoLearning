using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject
{
    public class PINN:Module
    {
        public PINN()
        {
            Layer = new Sequential(
                new Linear(1, 32),
                new Sigmoid(),
                new Linear(32, 32),
                new Sigmoid(),
                new Linear(32, 32),
                new Sigmoid(),
                new Linear(32, 1));
        }

        public Module Layer { get; set; }

        public override Tensor Forward(Tensor data)
        {
            return Layer.Forward(data);
        }
    }
}
