using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;

namespace TensorFlowProject
{
    public abstract class Module
    {
        public abstract Tensor Forward(Tensor data);
        public IVariableV1[] Parameters()
        {
            if(_parameters == null)
            {
                List<IVariableV1> variables = new List<IVariableV1>();
                //普通层数据
                var modules = GetType()
                    .GetProperties()
                    .Where(s => s.PropertyType.IsAssignableTo(typeof(Module)));
                foreach (var module in modules)
                    variables.AddRange(((Module)module.GetValue(this)!).Parameters());
                //集合层数据
                var arrays = GetType()
                    .GetProperties()
                    .Where(s => s.PropertyType == typeof(List<Module>));
                foreach (var array in arrays)
                {
                    var datas = (List<Module>)array.GetValue(this)!;
                    foreach (var data in datas)
                        variables.AddRange(data.Parameters());
                }
                //基本层数据
                var vars = GetType()
                    .GetProperties()
                    .Where(s => s.PropertyType == typeof(IVariableV1));
                foreach (var @var in vars)
                {
                    IVariableV1 data = (IVariableV1)var.GetValue(this)!;
                    variables.Add(data);
                }
                _parameters = variables.ToArray();
                return _parameters;
            }
            else
            {
                return _parameters;
            }
           
        }
        private IVariableV1[] _parameters { get; set; } = null;
    }
}
