using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;

namespace TensorFlowProject.Modules
{
    public class Sequential:Module
    {
        public List<Module> modules { get; } = new List<Module>();
        public Sequential(params Module[] models)
        {
            this.modules.AddRange(models);
        }

        public override Tensor Forward(Tensor data)
        {
            var @out = data;
            foreach (var module in modules)
            {
                @out = module.Forward(@out);
            }           
            return @out;
        }

     

    }
}
