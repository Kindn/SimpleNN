/* Copyright (c) 2021, Peiyan Liu
 *
 * filename: json.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include <data/json.hpp>

namespace sjson
{

    /******************************class JsonReader**************************/

    JsonReader::JsonReader()
    {
        src_str = "";
        src_str_len = 0;
        curr_pos = -1;
        curr_line = 0;
        curr_col = 0;
        eof_flag = true;
    }

    JsonReader::JsonReader(const std::string& _src_str)
    {
        src_str = _src_str;
        src_str_len = src_str.size();
        if (src_str_len <= 0)
        {
            curr_pos = -1;
            curr_line = 0;
            curr_col = 0;
                
            eof_flag = true;
        }
        else
        {
            curr_pos = 0;
            curr_line = 1;
            curr_col = 1;
            eof_flag = false;
        }
    }

    JsonReader::JsonReader(std::ifstream& _input_file_stream)
    {
        if (!_input_file_stream)
        {
            std::stringstream msg;
            msg << "In JsonReader::JsonReader: ";
            msg << "_input_file_path is null!" << std::endl;
            throw msg.str();
        }
        else
        {
            std::string line;
            
            while (std::getline(_input_file_stream, line))
            {
                this->src_str += (line + '\n');
            }

            //std::cout << src_str << std::endl;

            src_str_len = src_str.size();
            if (src_str_len <= 0)
            {
                curr_pos = -1;
                curr_line = 0;
                curr_col = 0;
                
                eof_flag = true;
            }
            else
            {
                curr_pos = 0;
                curr_line = 1;
                curr_col = 1;
                eof_flag = false;
            }
        }
    }

    char JsonReader::currChar() const
    {
        return src_str[curr_pos];
    }

    char JsonReader::getChar()
    {
        if (eof_flag)
        {
            throw JsonError("Json file ends unexpectedly!", 
                            getCurrLine(), getCurrCol());
        }
        char ch = src_str[curr_pos++];
        if (ch == '\n')
        {
            curr_line++;
            curr_col = 0;
        }
        else if (ch == '\t')
            curr_col += 4;
        else if (ch == '\r')
        {
            curr_col = 0;
        }
        else
            curr_col++;
        if (curr_pos == src_str_len)
            eof_flag = true;

        return ch;
    }

    std::string JsonReader::strAhead(int n) const
    {
        return src_str.substr(curr_pos, std::min<int>(n, 
                              src_str_len - curr_pos));
    }

    std::string JsonReader::getStr(int n)
    {
	    std::string str = strAhead(n);
        if (curr_pos >= src_str_len)
        {
            throw JsonError("Json file ends unexpectedly!", 
                            getCurrLine(), getCurrCol());
        }
	    curr_pos += n;
        return str;
    }

    int JsonReader::getCurrLine() const
    {
        return curr_line;
    }

    int JsonReader::getCurrCol() const
    {
        return curr_col;
    }

    void JsonReader::skip()
    {
        while (currChar() == ' ' ||
               currChar() == '\t' || 
               currChar() == '\r' ||
               currChar() == '\n')
        {
            getChar();
        }

        if (curr_pos == src_str_len)
            eof_flag = true;
    }

    bool JsonReader::isEOF()
    {
        return eof_flag;
    }

    /******************************class JsonNode_P*************************/

    JsonNode::JsonNode()
    {
        child_head = nullptr;
        next = nullptr;
        prev = nullptr;

        value_int = 0;
        value_double = 0;
        value_string = "";
        key = "";
        value_type = JSON_VALUE_TYPE_NULL;
        children_size = 0;
    }

    JsonNode::JsonNode(JsonValueType _value_type, 
                       const std::string& _key)
    {
        value_type = _value_type;
        key = _key;

        child_head = nullptr;
        next = nullptr;
        prev = nullptr;

        value_int = 0;
        value_double = 0;
        value_string = "";
        children_size = 0;
    }

    JsonNode::~JsonNode()
    {
        
    }

    int JsonNode::get_value_type() const
    {
        return value_type;
    }

    double JsonNode::get_number_value() const
    {
        if (value_type != JSON_VALUE_TYPE_INT && 
            value_type != JSON_VALUE_TYPE_DOUBLE )
        {
            std::stringstream msg;
            msg << "In JsonNode::get_value: ";
            msg << "value type mismatch!";
            throw msg.str();
        }
        else if (value_type == JSON_VALUE_TYPE_INT)
            return  double(value_int);
        else
            return  value_double;
    }

    bool JsonNode::get_bool_value() const
    {
        if (value_type != JSON_VALUE_TYPE_BOOL)
        {
            std::stringstream msg;
            msg << "In JsonNode::get_value: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
            return value_bool;
    }

    std::string JsonNode::get_string_value() const
    {
        if (value_type != JSON_VALUE_TYPE_STRING)
        {
            std::stringstream msg;
            msg << "In JsonNode::get_value: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
            return value_string;
    }

    JsonNode& JsonNode::get_object_value()
    {
        if (value_type != JSON_VALUE_TYPE_OBJECT)
        {
            std::stringstream msg;
            msg << "In JsonNode::get_value: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
            return *this;
    }

    int JsonNode::get_children_size() const
    {
        if (value_type != JSON_VALUE_TYPE_ARRAY && 
            value_type != JSON_VALUE_TYPE_OBJECT)
        {
            std::stringstream msg;
            msg << "In JsonNode::get_children_size: ";
            msg << "value type mismatch!";
            throw msg.str();
        }
        else
            return children_size;
    }

    void JsonNode::set_value(double _value_num)
    {
        if (value_type != JSON_VALUE_TYPE_INT && 
            value_type != JSON_VALUE_TYPE_DOUBLE && 
            value_type != JSON_VALUE_TYPE_BOOL)
        {
            std::stringstream msg;
            msg << "In JsonNode::get_value: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else if (value_type == JSON_VALUE_TYPE_INT)
            value_int = int(_value_num);
        else if (value_type == JSON_VALUE_TYPE_DOUBLE)
            value_double = _value_num;
        else
            value_bool = (_value_num > 0);
        
        children_size = 0;
    }


    void JsonNode::set_value(const std::string& _value_string)
    {
        if (value_type != JSON_VALUE_TYPE_STRING)
        {
            std::stringstream msg;
            msg << "In JsonNode::get_value: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
            value_string = _value_string;

        children_size = 0;
    }

    void JsonNode::set_key(const std::string& _key)
    {
        key = _key;
    }

    std::string JsonNode::get_key() const
    {
        return key;
    }

    void JsonNode::push_back(JsonNode_P _node)
    {
        if (value_type != JSON_VALUE_TYPE_ARRAY && 
            value_type != JSON_VALUE_TYPE_OBJECT)
        {
            std::stringstream msg;
            msg << "In JsonNode::array_push_back: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
        {
            std::shared_ptr<JsonNode> curr = child_head;
            if (curr == nullptr)
            {
                child_head = _node;
                child_head->next = nullptr;
                child_head->prev = nullptr;
                children_size++;
            }
            else
            {
                while (curr->next != nullptr)
                {
                    curr = curr->next;
                }
                curr->next = _node;
                curr->next->prev = curr;
                curr->next->next = nullptr;
                children_size++;
            }
        }   
    }

    void JsonNode::pop_back(JsonNode_P _node)
    {
        if (value_type != JSON_VALUE_TYPE_ARRAY && 
            value_type != JSON_VALUE_TYPE_OBJECT)
        {
            std::stringstream msg;
            msg << "In JsonNode::array_pop_back: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
        {
            JsonNode_P curr = child_head;
            if (curr == nullptr)
            {
                child_head = _node;
                child_head->next = nullptr;
                child_head->prev = nullptr;
                children_size++;
            }
            else
            {
                JsonNode_P second = child_head;
                child_head = _node;
                child_head->prev = nullptr;
                child_head->next = second;
                second->prev = child_head;
                children_size++;
            }
        }   
    }

    void JsonNode::clear()
    {
        child_head = nullptr;
        children_size = 0;
    }

    bool JsonNode::set_array(int _size)
    {
        if (value_type == JSON_VALUE_TYPE_NULL)
            setValueType(JSON_VALUE_TYPE_ARRAY);
        if (value_type != JSON_VALUE_TYPE_ARRAY)
            return false;
        try
        {
            clear();
            for (int i = 0; i < _size; i++)
            {
                JsonNode_P new_element(new JsonNode(JSON_VALUE_TYPE_NULL));
                push_back(new_element);
            }
            children_size = _size;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return false;
        }
        
        return true;
    }

    JsonNode& JsonNode::array_get(int _index)
    {
        if (value_type != JSON_VALUE_TYPE_ARRAY)
        {
            std::stringstream msg;
            msg << "In JsonNode::array_get: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
        {
            if (_index >= children_size || _index < 0)
            {
                std::stringstream msg;
                msg << "In JsonNode::array_get: ";
                msg << "array index out of bound!" << std::endl;
                throw msg.str();
            }
            else
            {
                JsonNode_P target = child_head;
                for (int i = 0; i < _index; i++)
                    target = target->next;

                return *target;
            }
        }
    }

    void JsonNode::array_insert(JsonNode_P _node, int _index)
    {
        if (value_type != JSON_VALUE_TYPE_ARRAY)
        {
            std::stringstream msg;
            msg << "In JsonNode::array_insert: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
        {
            if (_index > children_size || _index < 0)
            {
                std::stringstream msg;
                msg << "In JsonNode::array_insert: ";
                msg << "array index out of bound!" << std::endl;
                throw msg.str();
            }
            else
            {
                if (_index == 0)
                    pop_back(_node);
                else if (_index == children_size)
                    push_back(_node);
                else
                {
                    JsonNode_P curr = child_head;
                    for (int i = 0; i < _index - 1; i++)
                        curr = curr->next;
                    
                    JsonNode_P next_node = curr->next;
                    curr->next = _node;
                    curr->next->next = next_node;
                    curr->next->prev = curr;
                    next_node->prev = curr->next;
                    children_size++;
                }
            }
        }
    }

    void JsonNode::array_erase(int _index)
    {
        if (value_type != JSON_VALUE_TYPE_ARRAY)
        {
            std::stringstream msg;
            msg << "In JsonNode::array_remove: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
        {
            if (_index >= children_size || _index < 0)
            {
                std::stringstream msg;
                msg << "In JsonNode::array_remove: ";
                msg << "array index out of bound!" << std::endl;
                throw msg.str();
            }
            else
            {
                JsonNode_P target = child_head;
                for (int i = 0; i < _index; i++)
                    target = target->next;

                if (_index == 0)
                {
                    child_head = target->next;
                    if (children_size > 1)
                        child_head->prev = nullptr;
                    target->next = nullptr;
                    target->prev = nullptr;
                    children_size--;
                }
                else if (_index == children_size - 1)
                {
                    target->prev->next = nullptr;
                    target->prev = nullptr;
                    children_size--;
                }
                else
                {
                    target->prev->next = target->next;
                    target->next->prev = target->prev;
                    target->next = nullptr;
                    target->prev = nullptr;
                    children_size--;
                }
            }
        }
    }

    bool JsonNode::obj_has_item(const std::string& _key)
    {
        if (value_type != JSON_VALUE_TYPE_OBJECT)
        {
            std::stringstream msg;
            msg << "In JsonNode::obj_has_element: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
        {
            if (children_size <= 0)
                return false;
            else
            {
                bool has = false;
                JsonNode_P curr = child_head;
                for (int i = 0; i < children_size; i++)
                {
                    if (curr->key == _key)
                    {
                        has = true;
                        break;
                    }
                    curr = curr->next;
                }

                return has;
            }
        }
    }
    
    JsonNode& JsonNode::obj_get_item(const std::string& _key)
    {
        if (value_type != JSON_VALUE_TYPE_OBJECT)
        {
            std::stringstream msg;
            msg << "In JsonNode::obj_has_element: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
        {
            JsonNode_P curr = child_head;
            for (int i = 0; i < children_size; i++)
            {
                if (curr->key == _key)
                    return *curr;
                curr = curr->next;
            }
            
            throw JsonError("No item named " + _key, 
                            0, 
                            0, 
                            false);
        }
    }

    void JsonNode::obj_set_item(JsonNode_P _node)
    {
        if (value_type != JSON_VALUE_TYPE_OBJECT)
        {
            std::stringstream msg;
            msg << "In JsonNode::obj_set_element: ";
            msg << "value type mismatch!" << std::endl;
            throw msg.str();
        }
        else
        {
            JsonNode_P curr = child_head;
            for (int i = 0; i < children_size; i++)
            {
                if (curr->key == _node->key)
                {
                    if (i == 0)
                        child_head = _node;
                    else
                        curr->prev->next = _node;
                    if (i < children_size - 1)
                        curr->next->prev = _node;
                    _node->prev = curr->prev;
                    _node->next = curr->next;
                    curr->next = curr->prev = nullptr;

                    return;
                }
                curr = curr->next;
            }
            push_back(_node);
        }
    }

    void JsonNode::toJsonStr(std::string& _dst_str, 
                             bool _print_key, 
                             int _tab_num, 
                             bool _one_line_per_element) const
    {
        _dst_str.clear();
        for (int i = 0; i < _tab_num; i++)
            _dst_str += '\t';

        if (key.size() != 0 && _print_key)
            _dst_str += ('\"' + key + "\":");
        
        switch (value_type)
        {
        case JSON_VALUE_TYPE_INT:
        {
            std::string num_str = std::to_string(value_int);
            _dst_str += num_str;
        }
            break;
        
        case JSON_VALUE_TYPE_DOUBLE:
        {
            std::string num_str = std::to_string(value_double);
            _dst_str += num_str;
        }
            break;

        case JSON_VALUE_TYPE_BOOL:
        {
            if (value_bool)
                _dst_str += "true";
            else
                _dst_str += "false";
        }
            break;

        case JSON_VALUE_TYPE_STRING:
        {
            _dst_str += ('\"' + value_string + '\"');
        }
            break;

        case JSON_VALUE_TYPE_ARRAY:
        {
            _dst_str += '[';
            JsonNode_P curr = child_head;
            while (curr)
            {
                if (_one_line_per_element)
                {
                    _dst_str += '\n';
                    std::string element_str;
                    curr->toJsonStr(element_str, 
                                    false, 
                                    _tab_num + 1, 
                                    true);
                    _dst_str += element_str;
                }
                else
                {
                    std::string element_str;
                    curr->toJsonStr(element_str, 
                                    false, 
                                    0, 
                                    false);
                    _dst_str += element_str;
                }

                if (curr->next)
                    _dst_str += ',';
                else   // reach the end of array
                {
                    if (_one_line_per_element)
                    {
                        _dst_str += '\n';
                        for (int i = 0; i < _tab_num; i++)
                            _dst_str += '\t';
                        _dst_str += ']';
                    }
                    else
                        _dst_str += ']';
                    break;
                }

                curr = curr->next;
            }
            
            if(!curr) _dst_str += ']';
            break;
        }
            break;

        case JSON_VALUE_TYPE_OBJECT:
        {
            _dst_str += '{';
            JsonNode_P curr = child_head;
            while (curr)
            {
                if (_one_line_per_element)
                {
                    _dst_str += '\n';
                    std::string element_str;
                    curr->toJsonStr(element_str, 
                                    true, 
                                    _tab_num + 1, 
                                    true);
                    _dst_str += element_str;
                }
                else
                {
                    std::string element_str;
                    curr->toJsonStr(element_str, 
                                    true, 
                                    0, 
                                    false);
                    _dst_str += element_str;
                }

                if (curr->next)
                    _dst_str += ',';
                else   // reach the end of array
                {
                    if (_one_line_per_element)
                    {
                        _dst_str += '\n';
                        for (int i = 0; i < _tab_num; i++)
                            _dst_str += '\t';
                        _dst_str += '}';
                    }
                    else
                        _dst_str += '}';
                    break;
                }

                curr = curr->next;
            }
            
            if(!curr) _dst_str += '}';
            break;
        }
            break;

         case JSON_VALUE_TYPE_NULL:
        {
            _dst_str += "NULL";
        }
            break;

        default:
            break;
        }
        
    }

    JsonNode& JsonNode::operator [] (const std::string& _key)
    {
        if (value_type == JSON_VALUE_TYPE_NULL)
            setValueType(JSON_VALUE_TYPE_OBJECT);
        if (value_type != JSON_VALUE_TYPE_OBJECT)
            throw JsonError("Value type mismatched!", 
                            0, 
                            0, 
                            false);
        
        if (obj_has_item(_key))
            return obj_get_item(_key);
        else
        {
            JsonNode_P _new_item(new JsonNode(JSON_VALUE_TYPE_NULL, 
                                               _key));
            obj_set_item(_new_item);
            return *_new_item;
        }
    }

    JsonNode& JsonNode::operator [] (int _array_ind)
    {
        if (value_type == JSON_VALUE_TYPE_NULL)
            setValueType(JSON_VALUE_TYPE_ARRAY);
        if (value_type != JSON_VALUE_TYPE_ARRAY)
            throw JsonError("Value type mismatched!", 
                            0, 
                            0, 
                            false);

        if (_array_ind < 0 || _array_ind >= children_size)
            throw JsonError("Array index out of bound!", 
                            0, 
                            0, 
                            false);
        else
            return array_get(_array_ind);
    }

    double JsonNode::operator = (double _num_val)
    {
        if (value_type == JSON_VALUE_TYPE_NULL)
            setValueType(JSON_VALUE_TYPE_DOUBLE);
        if (value_type == JSON_VALUE_TYPE_DOUBLE)
        {
            value_double = _num_val;
            return _num_val;
        }
        else if (value_type == JSON_VALUE_TYPE_INT)
        {
            value_int = static_cast<int>(_num_val);
            return _num_val;
        }
        else if (value_type == JSON_VALUE_TYPE_BOOL)
        {
            value_bool = (_num_val > 0) ? true : false;
            return _num_val;
        }
        else
        {
            throw JsonError("Cannot pass double value to this node!", 
                            0, 
                            0, 
                            false);
        }
    }

    std::string JsonNode::operator = (const std::string& _str_val)
    {
        if (value_type == JSON_VALUE_TYPE_NULL)
            setValueType(JSON_VALUE_TYPE_STRING);
        if (value_type == JSON_VALUE_TYPE_STRING)
        {
            value_string = _str_val;
            return _str_val;
        }
        else
        {
            throw JsonError("Cannot pass string value to this node!", 
                            0, 
                            0, 
                            false);
        }
    }

    int JsonNode::as_int() const
    {
       switch (value_type)
       {
        case JSON_VALUE_TYPE_INT:
           return value_int;
           break;

        case JSON_VALUE_TYPE_DOUBLE:
           return static_cast<int>(value_double);
           break;

        case JSON_VALUE_TYPE_BOOL:
           return static_cast<int>(value_bool);
           break;

        case JSON_VALUE_TYPE_NULL:
           return 0;
           break;
       
        default:
            throw JsonError("Value is not convertible to Int!", 
                            0, 
                            0, 
                            false);
           break;
       } 
    }

    double JsonNode::as_double() const
    {
        switch (value_type)
       {
        case JSON_VALUE_TYPE_INT:
           return static_cast<double>(value_int);
           break;

        case JSON_VALUE_TYPE_DOUBLE:
           return value_double;
           break;

        case JSON_VALUE_TYPE_BOOL:
           return static_cast<double>(value_bool);
           break;

        case JSON_VALUE_TYPE_NULL:
           return 0;
           break;
       
        default:
            throw JsonError("Value is not convertible to Double!", 
                            0, 
                            0, 
                            false);
           break;
       } 
    }

    bool JsonNode::as_bool() const
    {
        switch (value_type)
        {
        case JSON_VALUE_TYPE_INT:
           return static_cast<bool>(value_int);
           break;

        case JSON_VALUE_TYPE_DOUBLE:
           return static_cast<bool>(value_double);
           break;

        case JSON_VALUE_TYPE_BOOL:
           return value_bool;
           break;

        case JSON_VALUE_TYPE_NULL:
           return false;
           break;
       
        default:
            throw JsonError("Value is not convertible to Bool!", 
                            0, 
                            0, 
                            false);
           break;
        }
    }

    std::string JsonNode::as_string() const
    {
        switch (value_type)
        {
        case JSON_VALUE_TYPE_STRING:
           return value_string;
           break;

        case JSON_VALUE_TYPE_INT:
           return std::to_string(value_int);
           break;

        case JSON_VALUE_TYPE_DOUBLE:
           return std::to_string(value_double);
           break;

        case JSON_VALUE_TYPE_BOOL:
           return value_bool ? "true" : "false";
           break;

        case JSON_VALUE_TYPE_NULL:
           return "NULL";
           break;
       
        default:
            throw JsonError("Value is not convertible to String!", 
                            0, 
                            0, 
                            false);
           break;
        }
    }

    std::vector<JsonNode_P> JsonNode::as_vector()
    {
        std::vector<JsonNode_P> vec;
        JsonNode_P curr_node = child_head;
        for (int i = 0; i < children_size; i++)
        {
            vec.push_back(curr_node);
            curr_node = curr_node->next;
        }

        return vec;
    }
            

    void JsonNode::setValueType(int _value_type)
    {
        value_type = _value_type;
    }

    /******************************class Json**************************/

    Json::Json()
    {
        root = std::make_shared<JsonNode>(JSON_VALUE_TYPE_OBJECT);
    }

    Json::Json(const std::string& _json_str)
    {
        root = std::make_shared<JsonNode>(JSON_VALUE_TYPE_OBJECT);
        try
        {
            JsonReader reader(_json_str);
            parseValue(root, reader);
            success = true;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            success = false;
        }  
    }

    Json::Json(std::ifstream& _ifs)
    {
        root = std::make_shared<JsonNode>(JSON_VALUE_TYPE_OBJECT);
        try
        {
            JsonReader reader(_ifs);
            parseValue(root, reader);
            success = true;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            success = false;
        }  
    }

    Json::~Json()
    {
        
    }

    JsonNode& Json::getRoot()
    {
        return *root;
    }

    bool Json::succeed() const
    {
        return success;
    }

    bool Json::fail() const
    {
        return !success;
    }
    
    void Json::parseValue(JsonNode_P _node, JsonReader& _reader)
    {
        _reader.skip();

        char ch = _reader.currChar();
        if (ch == '{')
            parseObject(_node, _reader);
        else if (ch == '\"')
            parseString(_node, _reader);
        else if (ch == '[')
            parseArray(_node, _reader);
        else if (ch == '-' || isdigit(ch))
            parseNumber(_node, _reader);
        else
        {
            if (_reader.strAhead(5) == "false")
            {
                _node->setValueType(JSON_VALUE_TYPE_BOOL);
                _node->set_value(0);
                _reader.getStr(5);
            }
            else if (_reader.strAhead(4) == "true")
            {
                _node->setValueType(JSON_VALUE_TYPE_BOOL);
                _node->set_value(1);
                _reader.getStr(4);
            }
            else if (_reader.strAhead(4) == "NULL")
            {
                _node->setValueType(JSON_VALUE_TYPE_NULL);
                _reader.getStr(4);
            }
            else
            {
                throw JsonError("Invalid Symbol:" + ch, 
                                _reader.getCurrLine(), 
                                _reader.getCurrCol());
            }
        }

    }

    void Json::parseNumber(JsonNode_P _node, JsonReader& _reader)
    {
        _node->setValueType(JSON_VALUE_TYPE_DOUBLE);
        double sign = 1;
        int scale = 0, sub_scale = 0;
        double value = 0;
        int exp_sign = 1;
        int exp_val = 0;

        char curr_char = _reader.currChar();
        if (curr_char != '-' && !isdigit(curr_char))
        {
            throw JsonError("Invalid Symbol:" + curr_char, 
                                _reader.getCurrLine(), 
                                _reader.getCurrCol());
        }
        if (curr_char == '-')
        {
            sign = -1;
            _reader.getChar();
            curr_char = _reader.currChar();
        }
        
        if (isdigit(curr_char) && curr_char != '0')
        {
            while (isdigit(curr_char))
            {
                value = value * 10 + double(curr_char - '0');
                _reader.getChar();
                curr_char = _reader.currChar();
            }
        }
        else if (curr_char == '0')
        {
            _reader.getChar();
            curr_char = _reader.currChar();
        }
            
        if (curr_char == '.')
        {
            _reader.getChar();
            curr_char = _reader.currChar();
            if (!isdigit(curr_char))
            {
                throw JsonError("Invalid Symbol:" + curr_char, 
                                _reader.getCurrLine(), 
                                _reader.getCurrCol());
            }
            else
            {
                while (isdigit(curr_char))
                {
                    sub_scale += 1;
                    value = value + double(curr_char - '0') * 
                                    std::pow(0.1, double(sub_scale));
                    _reader.getChar();
                    curr_char = _reader.currChar();
                }
            }
        }

        if (curr_char == 'e' || curr_char == 'E')
        {
            _reader.getChar();
            curr_char = _reader.currChar();

            if (curr_char == '-')
            {
                exp_sign = -1;
                _reader.getChar();
                curr_char = _reader.currChar();
            }
            else if (curr_char == '+')
            {
                _reader.getChar();
                curr_char = _reader.currChar();
            }

            if (!isdigit(curr_char))
            {
                throw JsonError("Invalid Symbol:" + curr_char, 
                                _reader.getCurrLine(), 
                                _reader.getCurrCol());
            }
            else
            {
                while (isdigit(curr_char))
                {
                    exp_val = exp_val * 10 + (curr_char - '0');
                    _reader.getChar();
                    curr_char = _reader.currChar();
                }
            }

            exp_val *= exp_sign;
        }

        value = (value * std::pow(10, double(exp_val))) * sign;
        _node->set_value(value);
    }

    void Json::parseString(JsonNode_P _node, JsonReader& _reader)
    {
        _node->setValueType(JSON_VALUE_TYPE_STRING);
        std::string str;

        char curr_char = _reader.currChar();
        if (curr_char != '\"')
        {
            throw JsonError("Invalid string start symbol:" + curr_char, 
                            _reader.getCurrLine(), 
                            _reader.getCurrCol());
        }
        else
        {
            _reader.getChar();
            curr_char = _reader.currChar();
        }

        while (curr_char != '\"')
        {
            if (curr_char == '\\') // escaped character
            {
                _reader.getChar();
                curr_char = _reader.currChar();
                if (curr_char == '\"' || curr_char == '\\' || 
                    curr_char == '/' || curr_char == '\b' || 
                    curr_char == '\f' || curr_char == '\n' || 
                    curr_char == '\r' || curr_char =='\t')
                    str += curr_char;
                else if (curr_char == 'u') // unicode character \uxxxx
                {
                    std::string unicode_str; 
                    for (int i = 0; i < 4; i++)
                    {
                        _reader.getChar();
                        curr_char = _reader.currChar();
                        if ((curr_char >= '0' && curr_char <= '9') || 
                            (curr_char >= 'a' && curr_char <= 'f') || 
                            (curr_char >= 'A' && curr_char <= 'F'))
                            unicode_str += curr_char;
                        else
                        {
                            throw JsonError("Invalid unicode string symbol: " + 
                                                                    curr_char, 
                                            _reader.getCurrLine(), 
                                            _reader.getCurrCol());
                        }
                    }
                    
                    long unicode = strtol(unicode_str.data(), nullptr, 16);
                    std::string utf8_out;
                    encode_utf8(unicode, utf8_out);
                    str += utf8_out;
                }
                else
                {
                    throw JsonError("Invalid escaped character: " + '\\' + curr_char, 
                                    _reader.getCurrLine(), 
                                    _reader.getCurrCol());
                }
                _reader.getChar();
                curr_char = _reader.currChar();
            }
            else
            {
                str += curr_char;
                _reader.getChar();
                curr_char = _reader.currChar();
            }
        }

        _reader.getChar();   // get the '\"' at the end of string
        _node->set_value(str);
    }

    void Json::parseArray(JsonNode_P _node, JsonReader& _reader)
    {
        _node->setValueType(JSON_VALUE_TYPE_ARRAY);
        char curr_char = _reader.currChar();
        if (curr_char != '[')
        {
            throw JsonError("Invalid string start symbol:" + curr_char, 
                            _reader.getCurrLine(), 
                            _reader.getCurrCol());
        }
        else
        {
            _reader.getChar();
            _reader.skip();
            curr_char = _reader.currChar();
        }

        while (curr_char != ']')
        {
            JsonNode_P element(new JsonNode);
            parseValue(element, _reader);
            _node->push_back(element);

            _reader.skip();
            curr_char = _reader.currChar();
            if (curr_char == ',')
            {
                _reader.getChar();
                _reader.skip();
            }
            else if (curr_char != ']')
            {
                throw JsonError("Invalid symbol:" + curr_char, 
                                _reader.getCurrLine(), 
                                _reader.getCurrCol());
            }
            else
            {
                _reader.getChar();
                return;
            }
            curr_char = _reader.currChar();
        }

        _reader.getChar();
    }

    void Json::parseObject(JsonNode_P _node, JsonReader& _reader)
    {
        _node->setValueType(JSON_VALUE_TYPE_OBJECT);
        char curr_char = _reader.currChar();
        if (curr_char != '{')
        {
            throw JsonError("Invalid object start symbol:" + curr_char, 
                            _reader.getCurrLine(), 
                            _reader.getCurrCol());
        }
        else
        {
            _reader.getChar();
            _reader.skip();
            curr_char = _reader.currChar();
        }

        while (curr_char != '}')
        {
            if (curr_char != '\"')
            {
                throw JsonError("Invalid symbol:" + curr_char, 
                                _reader.getCurrLine(), 
                                _reader.getCurrCol());
            }
            else
            {
                _reader.getChar();
                curr_char = _reader.currChar();
                std::string key;
                while (curr_char != '\"')
                {
                    key += _reader.getChar();
                    curr_char = _reader.currChar();
                }
                _reader.getChar();
                _reader.skip();
                curr_char = _reader.currChar();

                if(curr_char != ':')
                {
                    throw JsonError("Invalid symbol:" + curr_char, 
                                _reader.getCurrLine(), 
                                _reader.getCurrCol());
                }
                else
                {
                    _reader.getChar();
                    _reader.skip();
                    curr_char = _reader.currChar();
                    JsonNode_P item(new JsonNode(JSON_VALUE_TYPE_NULL, key));
                    parseValue(item, _reader);
                    _node->push_back(item);

                    _reader.skip();
                    curr_char = _reader.currChar();

                    if (curr_char == ',')
                    {
                        _reader.getChar();
                        _reader.skip();
                    }
                    else if (curr_char != '}')
                    {
                        throw JsonError("Invalid symbol:" + curr_char, 
                                        _reader.getCurrLine(), 
                                        _reader.getCurrCol());
                    }
                    else
                    {
                        _reader.getChar();
                        return;
                    }
                }
            }
            curr_char = _reader.currChar();
        }

        _reader.getChar();
    }

    void Json::toString(std::string& _output_str, 
                        bool _one_line_per_item) const
    {
        _output_str.clear();
        root->toJsonStr(_output_str, false, 0, _one_line_per_item);
    }

    bool Json::save(const std::string& _output_file_path, 
                    bool _one_line_per_item) const
    {
        std::ofstream ofs(_output_file_path);
        if (ofs.fail())
            return false;

        std::string output_str;
        toString(output_str, _one_line_per_item);
        ofs << output_str;

        return true;
    }

    /**************************other functions****************************/

    void encode_utf8(long _unicode, std::string& _out)
    {
        if (_unicode < 0 || _unicode > 0x10FFFF)
            return;

        _out.clear();
        if (_unicode <= 0x7F)
        {
            _out += static_cast<char>(_unicode);
        }
        else if (_unicode >= 0x80 && _unicode <= 0x7FF)
        {
            _out += static_cast<char>((_unicode >> 6) | 0xC0);
            _out += static_cast<char>((_unicode & 0x3F) | 0x80);
        }
        else if (_unicode >= 0x800 && _unicode <= 0xFFFF)
        {
            _out += static_cast<char>((_unicode >> 12) | 0xE0);
            _out += static_cast<char>(((_unicode >> 6) & 0x3F) | 0x80);
            _out += static_cast<char>((_unicode & 0x3F) | 0x80);
        }
        else if (_unicode >= 0x10000 && _unicode <= 0x10FFFF)
        {
            _out += static_cast<char>((_unicode >> 18) | 0xF0);
            _out += static_cast<char>(((_unicode >> 12) & 0x3F) | 0x80);
            _out += static_cast<char>(((_unicode >> 6) & 0x3F) | 0x80);
            _out += static_cast<char>((_unicode & 0x3F) | 0x80);
        }
        
    }

} // namespace sjson

