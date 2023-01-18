using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowProject
{
    public static class Operator
    {
        public static Tensor Exp(Tensor tensor)
        {
            return tf.exp(tensor);
        }

        public static Tensor Log(Tensor tensor) 
        {
            return tf.log(tensor);
        }

        public static Tensor Uniform(float min, float max, Shape shape)
        {
            return tf.random_uniform(shape, min, max);
        }

        public static Tensor Uniform(float min, float max, params int[] shapes)
        {
            return tf.random_uniform(new Shape(shapes), min, max);
        }

        public static Tensor Cast(Tensor x, TF_DataType dtype)
        {
            return tf.cast(x, dtype);
        }

        public static Tensor Equal(Tensor x, Tensor y)
        {
            return tf.equal(x, y);
        }

        public static Tensor Pow<T1, T2>(T1 x, T2 y) 
        {
            return tf.pow(x, y);
        }

        public static Tensor Zero(Shape shape)
        {
            return tf.zeros(shape);
        }

        public static Tensor Zero(params int[] shape) 
        {
            return tf.zeros(new Shape(shape));
        }

        public static Tensor ReduceMean(Tensor x)
        {
            return tf.reduce_mean(x);
        }

        public static Tensor Sum(Tensor x, int axis)
        {
            return tf.sum(x, axis);
        }
    }
}
