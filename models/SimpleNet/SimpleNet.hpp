#include <../include/SimpleNN.hpp>

namespace snn
{
    class SimpleNet: public Sequential
    {
        public:
            SimpleNet():Sequential()
            {
                std::shared_ptr<Affine> aff1_layer(new Affine(28 * 28, 1, 512, 1));
                std::shared_ptr<Relu> relu_layer(new Relu);
                std::shared_ptr<Affine> aff2_layer(new Affine(512, 1, 10, 1));
                std::shared_ptr<Softmax> softmax_layer(new Softmax);

                *this << aff1_layer << relu_layer << aff2_layer << softmax_layer;
            }
    };
}
