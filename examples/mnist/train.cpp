#include <SimpleNN.hpp>
#include <LeNet/LeNet.hpp>

using namespace snn;

const std::string train_data_file_path = "../../../examples/mnist/dataset/images_train.csv";
const std::string test_data_file_path = "../../../examples/mnist/dataset/images_test.csv";
const std::string train_label_file_path = "../../../examples/mnist/dataset/labels_train.csv";
const std::string test_label_file_path = "../../../examples/mnist/dataset/labels_test.csv";

int main()
{
    LeNet net;
    
    std::shared_ptr<SoftmaxWithLoss> loss_layer(new SoftmaxWithLoss(true));
    net.set_loss_layer(loss_layer);
    std::cout << "Loss layer ready!" << std::endl;

    std::vector<Matrix_d> init_params = net.get_params();
    std::vector<Matrix_d> init_grads = net.get_grads();
    std::shared_ptr<AdaGrad> opt(new AdaGrad(init_params, init_grads, 0.012));
    //std::shared_ptr<GradDescent> opt(new GradDescent(init_params, init_grads, 0.004));
    //std::shared_ptr<Momentum> opt(new Momentum(init_params, init_grads, 0.6, 2));
    net.set_optimizer(opt);
    std::cout << "Optimizer ready!" << std::endl;
    
    Dataset train_set(true);
    Dataset test_set(true);
    
    if (train_set.load_data(train_data_file_path, train_label_file_path))
        std::cout << "Train set loading finished!" << std::endl;
    else
        std::cout << "Failed to load train set data!" << std::endl;

    if (test_set.load_data(test_data_file_path, test_label_file_path))
        std::cout << "Test set loading finished!" << std::endl;
    else
        std::cout << "Failed to load test set data!" << std::endl;

    net.fit(train_set, test_set, 256, 2);

    if (!net.save_net("../../../examples/mnist/LeNet.net"))
    {
        std::cout << "Failed to save net!" << std::endl;
        return 0;
    }
    if (!net.save_weights("../../../examples/mnist/LeNet.weights"))
    {
        std::cout << "Failed to save weights!" << std::endl;
        return 0;
    }

    return 0;
}
