using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Optimizer = TensorFlowProject.Optimizers.Optimizer;

namespace TensorFlowProject.Criterions
{
    public class Loss
    {
        public Loss() { }
        public Loss(Tensor value)
        {
            Value = value;
        }
        public Tensor Value { get; set; }
        public void Backward() 
        {
            Optimizer.Gradients = Optimizer.Tape.gradient(Value, Optimizer.Parameters);
        }

        public override string ToString()
        {
            return Value.ToString();
        }

        public static Loss operator +(Loss a, Loss b)
            => new Loss(a.Value + b.Value);

        public static Loss operator -(Loss a, Loss b)
            => new Loss(a.Value - b.Value);

        public static implicit operator float(Loss loss)
            =>(float)loss.Value;
    }
}
