using System;
using TensorFlowProject.Criterions;
using TensorFlowProject.Optimizers;
using Adam = TensorFlowProject.Optimizers.Adam;
using Optimizer = TensorFlowProject.Optimizers.Optimizer;
using SGD = TensorFlowProject.Optimizers.SGD;

Demo3 NN = new();
NN.Start();


/// <summary>
/// MNIST图片分类
/// </summary>
class Demo1
{
    public void Start()
    {
        // 超参数
        float learning_rate = 0.001f;// 学习率
        int training_steps = 1000;// 训练轮数
        int batch_size = 256;// 批次大小
        int display_step = 100;// 训练数据 显示周期
        var ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();
        //数据归一化(将像素值(0-255)压缩到(0-1))
        (x_train, x_test) = (x_train / 255f, x_test / 255f);
        //把数据集放入数据迭代单元
        var train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);//转换为 Dataset 格式
        train_data = train_data.repeat()
            .shuffle(5000)
            .batch(batch_size)
            .prefetch(1)
            .take(training_steps);// 数据预处理

        CNN model = new CNN();
        Console.WriteLine(model.Parameters().Count());
        Criterion criterion = new CrossEntropyLoss();
        // 采用随机梯度下降优化器
        var optimizer = new SGD(model.Parameters(), 0.001f);

        foreach (var epoch in Enumerable.Range(0, 1000))
        {
            //训练网络
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                optimizer.ZeroGrad();   //刷新记录的梯度                
                var @out = model.Forward(batch_x);  //前馈获得输出
                var loss = criterion.Forward(@out, batch_y);    //获得损失
                loss.Backward();    //反向传播累积梯度
                optimizer.Step();   //优化参数

                if (step % 100 == 0)
                {
                    var pred = model.Forward(batch_x);
                    var accuracy = (float)Accuracy(pred, batch_y);
                    Console.WriteLine($"Epoch:{epoch},Train Accuracy: {accuracy},loss:{(float)loss}");
                }
            }
        }


        Tensor Accuracy(Tensor y_pred, Tensor y_true)
        {
            var max = tf.arg_max(y_pred, 1);
            var exact = tf.cast(y_true, tf.int64);
            var result = tf.reduce_mean(tf.cast(tf.equal(max, exact), tf.float32), axis: -1);
            return result;
        }
    }
}

class Demo2
{
    public void Start()
    {
        float lr = 0.001f; //学习率
        Module model = new PINN();
        Optimizer optimizer = new Adam(model.Parameters(), lr);
        Criterion criterion = new MSELoss();


        foreach (var epoch in Enumerable.Range(1, 10000))
        {
            //生成数据
            var x = Operator.Uniform(0, 1, 2000, 1);
            optimizer.ZeroGrad();
            var y = model.Forward(x);
            //产生约束
            y = y - Operator.Pow(x, 2) - 3 * x; // y-x^2-3x
            var loss = criterion.Forward(y, Operator.Zero(2000, 1));
            //反向传播
            loss.Backward();
            optimizer.Step();
            if(epoch % 1000 == 0) 
            {
                x = Operator.Uniform(0, 1, 2000, 1);
                y = model.Forward(x);
                var exact = Operator.Pow(x, 2) + 3 * x;
                var val = Operator.ReduceMean(exact - y);
                Console.WriteLine($"epoch:{epoch},loss:{(float)loss},std:{(float)val}");
            }
        }
        
    }
}

class Demo3
{
    public void Start()
    {
        float lr = 0.001f; //学习率
        Module model = new DNN();
        Optimizer optimizer = new Adam(model.Parameters(), lr);
        Criterion criterion = new CrossEntropyLoss();


        var (x_train, y_train) = GetTrainData();
        var (x_test, y_test) = GetTestData();

        var max = tf.max(tf.max(x_train, -1),-1);
        var min = tf.min(tf.min(x_train, -1),-1);
        x_train = (x_train - min) / (max - min);

        max = tf.max(tf.max(x_test, -1),-1);
        min = tf.min(tf.min(x_test, -1),-1);
        x_test = (x_test - min) / (max - min);
       

        //载入数据
        foreach (var epoch in Enumerable.Range(1, 10000))
        {
            optimizer.ZeroGrad();
            var predict = model.Forward(x_train);
            var loss = criterion.Forward(predict, y_train);
            //反向传播
            loss.Backward();
            optimizer.Step();
            if (epoch % 1000 == 0)
            {
                predict = model.Forward(x_test);
                var accuracy = Accuracy(predict, y_test);
                Console.WriteLine($"epoch:{epoch},loss:{(float)loss},accuracy:{(float)accuracy}");
            }
        }

    }

    Tensor Accuracy(Tensor y_pred, Tensor y_true)
    {
        var max = tf.arg_max(y_pred, 1);
        var exact = tf.cast(y_true, tf.int64);
        var result = tf.reduce_mean(tf.cast(tf.equal(max, exact), tf.float32), axis: -1);
        return result;
    }


    public (Tensor, Tensor) GetTrainData()
    {
        var excel = new Excel(false, "dataset.xlsx");
        var datas = new double[1400, 2];
        var target = new int[1400];
        for (int i = 0; i < 1400; ++i)
        {
            datas[i, 0] = Convert.ToDouble(excel[i, 1]);
            datas[i, 1] = Convert.ToDouble(excel[i, 2]);
            target[i] = Convert.ToInt32(excel[i, 0]);
        }
        return (tf.convert_to_tensor(datas, TF_DataType.TF_FLOAT), tf.convert_to_tensor(target, TF_DataType.TF_INT64));
    }

    public (Tensor, Tensor) GetTestData()
    {
        var excel = new Excel(false, "dataset.xlsx");
        var datas = new double[600, 2];
        var target = new int[600];
        for (int i = 1400; i < 2000; ++i)
        {
            datas[i - 1400, 0] = Convert.ToDouble(excel[i, 1]);
            datas[i - 1400, 1] = Convert.ToDouble(excel[i, 2]);
            target[i - 1400] = Convert.ToInt32(excel[i, 0]);
        }
        return (tf.convert_to_tensor(datas, TF_DataType.TF_FLOAT), tf.convert_to_tensor(target, TF_DataType.TF_INT64));
    }
}

