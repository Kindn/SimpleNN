/*
 * filename: Layer.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    Declaration of basic class Layer.
 */
#ifndef  _LAYERS_HPP_
#define  _LAYERS_HPP_

#include <Matrix/Matrix.hpp>
#include <functions/functions.hpp>
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace snn
{
    typedef std::map<const std::string, double>    LayerParams;

    class Layer : public std::enable_shared_from_this<Layer>
    {
        public:
            int input_rows, input_cols;
            int output_rows, output_cols;
            //int din_rows, din_cols;

            Matrix_d input_data, output_data;
            Matrix_d din;

        public:
            Layer();
            Layer(int _input_rows, int _input_cols,
                int _output_rows, int _output_cols);
            virtual ~Layer() {}
            
            void set_input(const Matrix_d _input_data)
            {
                input_data = _input_data;
            }

            Matrix_d get_output()
            {
                return output_data;
            }

            virtual bool set_properties(LayerParams& _layer_params) = 0;
            virtual bool get_properties(LayerParams& _layer_params, 
                                        std::string& _layer_type_got) = 0;
            virtual void get_params_ptr(std::vector<Matrix_d*>& _params, std::vector<Matrix_d*>& _grads) = 0;

            virtual void forward() = 0;
            virtual void backward(Matrix_d& dout) = 0;

    };

}


#endif // _LAYERS_HPP_
