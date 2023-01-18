using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject.Modules
{
    public class Dropout: Module
    {
        public Dropout(double rate=0.5) 
        {
            this.rate = (float)rate;
        }

        private float rate { get; set; }
        public override Tensor Forward(Tensor data)
        {
            return keras.layers.Dropout(rate).Apply(data);
        }
    }
}
