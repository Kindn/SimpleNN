#ifndef  _EXCEPTIONS_HPP_
#define  _EXCEPTIONS__HPP_

#include <iostream>
#include <string>
#include <sstream>
#include <exception>

namespace sjson
{
    class JsonError : public std::exception
    {
        private:
            std::stringstream msg;
        public:
            JsonError(const std::string &_description, 
                      int _line, 
                      int _col, 
                      bool _print_pos = true)
            {
                msg << "[JsonError]";
                msg << _description;
                if (_print_pos) 
                    msg << "(line " << _line << " col " << _col << "\n";
            }
            
            const char * what () const throw ()
            {
                return msg.str().c_str();
            }
    };
} // namespace sjson



#endif //_EXCEPTIONS__HPP_
