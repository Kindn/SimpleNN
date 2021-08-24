/* Copyright (c) 2021, Peiyan Liu
 *
 * filename: json.hpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef  _JSON_HPP_
#define  _JSON_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <sstream>
#include <math.h>
#include <data/exceptions.hpp>

namespace sjson
{

    class JsonReader;
    class JsonNode;
    class Json;

    typedef std::shared_ptr<JsonNode>  JsonNode_P;

    enum JsonValueType
    {
        JSON_VALUE_TYPE_INT = 0, 
        JSON_VALUE_TYPE_DOUBLE, 
        JSON_VALUE_TYPE_BOOL, 
        JSON_VALUE_TYPE_STRING, 
        JSON_VALUE_TYPE_ARRAY, 
        JSON_VALUE_TYPE_OBJECT,
        JSON_VALUE_TYPE_NULL
    };

    class JsonReader : public std::enable_shared_from_this<JsonReader>
    {
        private:
            std::string src_str;

            int src_str_len;
            int curr_pos;
            int curr_line, curr_col;

            int eof_flag;

        public:
            JsonReader();
            JsonReader(const std::string& _src_str);
            JsonReader(std::ifstream& _input_file_stream);

            ~JsonReader() {}

        public:
            char currChar() const;
            char getChar();
            std::string strAhead(int n) const;
            std::string getStr(int n);
            int getCurrLine() const;
            int getCurrCol() const;

            void skip();

            bool isEOF();
    };



    class JsonNode : public std::enable_shared_from_this<JsonNode>
    {
        private:
            JsonNode_P child_head;
            JsonNode_P next; 
            JsonNode_P prev;

            int       value_int;
            double    value_double;
            bool      value_bool;
            std::string value_string;

        private:
            std::string key;
            int         value_type;

            int         children_size;
        
        public:
            JsonNode();
            JsonNode(JsonValueType _value_type, 
                     const std::string& _key = "");

            ~JsonNode();

        public:
            int          get_value_type() const;
        
            double       get_number_value() const;
            bool         get_bool_value() const;
            std::string  get_string_value() const;
            JsonNode&    get_object_value();

            int          get_children_size() const;

            void         set_value(double _value_num);
            void         set_value(const std::string& _value_string);

            void         set_key(const std::string& _key);
            std::string  get_key() const;

        public:
            void push_back(JsonNode_P _node);
            void pop_back(JsonNode_P _node);
            void clear();

        public:
            bool      set_array(int _size = 0);
            JsonNode& array_get(int _index);
            void      array_insert(JsonNode_P _node, int _index);
            void      array_erase(int _index);

            bool      obj_has_item(const std::string& _key);
            JsonNode& obj_get_item(const std::string& _key);
            void      obj_set_item(JsonNode_P _node);
            
        public:
            void toJsonStr(std::string& _dst_str, 
                           bool _print_key = true, 
                           int _tab_num = 0, 
                           bool _one_line_per_element = false) const;

            JsonNode& operator [] (const std::string& _key);
            JsonNode& operator [] (int _array_ind);

            double      operator = (double _num_val);
            std::string operator = (const std::string& _str_val);

            int         as_int() const;
            double      as_double() const;
            bool        as_bool() const;
            std::string as_string() const;
            std::vector<JsonNode_P> as_vector();

        private:
            void setValueType(int _value_type);

        friend class Json;
    };

    class Json
    {
        private:
            JsonNode_P root;

            bool success;

        public:
            Json();
            Json(const std::string& _json_str);
            Json(std::ifstream& _ifs);
            ~Json();
        
        public:
            JsonNode& getRoot();
            bool succeed() const;
            bool fail() const;
        private: // for parsing
            void parseValue(JsonNode_P _node, JsonReader& _reader);
            void parseNumber(JsonNode_P _node, JsonReader& _reader);
            void parseString(JsonNode_P _node, JsonReader& _reader);
            void parseArray(JsonNode_P _node, JsonReader& _reader);
            void parseObject(JsonNode_P _node, JsonReader& _reader);

        public:
            void toString(std::string& _output_str, 
                          bool _one_line_per_item = false) const;
            bool save(const std::string& _output_file_path, 
                      bool _one_line_per_item = false) const;

    };


    void encode_utf8(long _unicode, std::string& _out);

} // namespace sjson

#endif //_JSON_HPP_
