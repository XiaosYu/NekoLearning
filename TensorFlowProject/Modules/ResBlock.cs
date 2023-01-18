using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject.Modules
{
    public class ResBlock : Module
    {
        public ResBlock(Module hidden, Module active)
        {
            this.Hidden = hidden;
            this.Active = active;
        }
        public Module Hidden { get; }
        public Module Active { get; }
        public override Tensor Forward(Tensor data)
        {
            var @out = this.Hidden.Forward(data);
            @out = this.Active.Forward(@out + data);
            return @out;
        }
    }
}
