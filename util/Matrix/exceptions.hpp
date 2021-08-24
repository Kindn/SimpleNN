#ifndef  _ILLEGALPARAMETERVALUE_HPP_
#define  _ILLEGALPARAMETERVALUE_HPP_

#include <iostream>
#include <string>

class illegalParameterValue
{
    public:
        illegalParameterValue():
            message("Illegal parameter value!") {}
        illegalParameterValue(const char* msg)
            {this->message = msg;}
        void outputMessage() {std::cout << message << std::endl;}

    private:
        std::string message;
};

class divisionByZero
{
    public:
        divisionByZero():
            message("Division by zero!") {}
        divisionByZero(const char* msg)
            {this->message = msg;}
        void outputMessage() {std::cout << message << std::endl;}

    private:
        std::string message;
};

class matrixIndexOutOfBound
{
    public:
        matrixIndexOutOfBound():
            message("Matrix index out of bound!") {}
        matrixIndexOutOfBound(const char* msg)
            {this->message = msg;}
        void outputMessage() {std::cout << message << std::endl;}

    private:
        std::string message;
};

class matrixSizeMismatch
{
    public:
        matrixSizeMismatch():
            message("Matrix size mismatched!") {}
        matrixSizeMismatch(const char* msg)
            {this->message = msg;}
        void outputMessage() {std::cout << message << std::endl;}

    private:
        std::string message;
};

class matrixIsSingular
{
    public:
        matrixIsSingular():
            message("Matrix is singular!") {}
        matrixIsSingular(const char* msg)
            {this->message = msg;}
        void outputMessage() {std::cout << message << std::endl;}

    private:
        std::string message;
};

#endif //_ILLEGALPARAMETERVALUE_HPP_
