#pragma once

#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <string>
#include <variant>
#include "midi.pb.h"

namespace encoder {

    using TOKEN_VARIANT = std::variant<int,std::string,std::tuple<int,int>>; // The different value types that a token value can have
    using TOKEN_TUPLE = std::tuple<midi::TOKEN_TYPE,TOKEN_VARIANT>; // A complete token in represented by its token type and token value

    struct INT_RANGE_FLAG {};
    struct INT_VALUES_FLAG {};
    struct INT_MAP_FLAG {};
    struct STRING_VALUES_FLAG {};
    struct STRING_MAP_FLAG {};
    struct TIMESIG_VALUES_FLAG {};
    struct TIMESIG_MAP_FLAG {};
    struct CONTINUOUS_FLAG {};
    struct STRING_VECTOR_FLAG {};

    INT_RANGE_FLAG RANGE_DOMAIN;
    INT_VALUES_FLAG INT_VALUES_DOMAIN;
    INT_MAP_FLAG INT_MAP_DOMAIN;
    STRING_VALUES_FLAG STRING_VALUES_DOMAIN;
    STRING_MAP_FLAG STRING_MAP_DOMAIN;
    TIMESIG_VALUES_FLAG TIMESIG_VALUES_DOMAIN;
    TIMESIG_MAP_FLAG TIMESIG_MAP_DOMAIN;
    CONTINUOUS_FLAG CONTINUOUS_DOMAIN;
    STRING_VECTOR_FLAG STRING_VECTOR;

    // different representation for an input token
    enum TOKEN_INPUT_TYPE {
        TI_INT,
        TI_STRING,
        TI_TIMESIG
    };

    class TOKEN_DOMAIN {

    /*
    Represents the domain of a token type. It encodes a token by mapping it input type (TOKEN_INPUT_TYPE) to a unique integer.
    Either it automatically increments this unique integer when a new token value is added, or you can provide a custom mapping (in this case contiguous_ouput is used to ensure a contiguous output domain, ie. [1,m] with some m)
    This integer is unique within the given domain. The representation.h class then creates unique ids from multiple token domains
    */

    public:

        TOKEN_DOMAIN(size_t n) { // for token types that take values in [0,n]
            if (n > 512) {
                throw std::invalid_argument("TOKEN DOMAIN SIZE IS TOO LARGE!");
            }
            for (int value=0; value<(int)n; value++) {
                add(value);
                input_types.push_back( TI_INT );
            }
        }
        TOKEN_DOMAIN(int min, int max, INT_RANGE_FLAG x) { // for token types that take values in [min,max[
            for (int value=min; value<max; value++) {
                add(value);
                input_types.push_back( TI_INT );
            }
        }
        TOKEN_DOMAIN(std::vector<int> values, INT_VALUES_FLAG x) { // for a custom set of int values
            for (const auto &value : values) {
                add(value);
                input_types.push_back( TI_INT );
            }
        }
        TOKEN_DOMAIN(std::map<int,int> values, INT_MAP_FLAG x) { // for a custom set of input int to output mappings
            for (const auto &kv : values) {
                add(kv.first, kv.second);
                input_types.push_back( TI_INT );
            }
        }
        TOKEN_DOMAIN(std::vector<std::string> values, STRING_VALUES_FLAG x) { // for a custom set of string values
            for (const auto &value : values) {
                add(value);
                input_types.push_back( TI_STRING );
            }
        }
        TOKEN_DOMAIN(std::map<std::string,int> values, STRING_MAP_FLAG x) { // for a custom set of input string to output mappings
            for (const auto &kv : values) {
                add(kv.first, kv.second);
                input_types.push_back( TI_STRING );
            }
        }
        TOKEN_DOMAIN(std::vector<std::tuple<int,int>> values, TIMESIG_VALUES_FLAG x) { // for a custom set of int pairs (time signatures)
            for (const auto &value : values) {
                add(value);
                input_types.push_back( TI_TIMESIG );
            }
        }
        TOKEN_DOMAIN(std::map<std::tuple<int,int>,int> values, TIMESIG_MAP_FLAG x) { // for a custom set of input int pairs (time signatures) to output mappings
            for (const auto &kv : values) {
                add(kv.first, kv.second);
                input_types.push_back( TI_TIMESIG );
            }
        }
        TOKEN_DOMAIN(int n, midi::TOKEN_TYPE rtt) {
            for (int value=0; value<n; value++) {
                add(value);
                input_types.push_back( TI_INT );
            }
            repeat_tt.push_back(rtt);
        }
        TOKEN_DOMAIN(std::vector<float> values, CONTINUOUS_FLAG x) {
            throw std::runtime_error("NOT CURRENTLY IMPLEMENTED!");
        }
        void add_internal(TOKEN_VARIANT x, int y) {
            // map token input x to token output y and respectively add them to their domains
            map_items.push_back( std::make_tuple(x,y) );
            mapping.insert( std::make_pair(x,y) );
            output_domain.insert( y );
            input_domain.insert( x );
        }
        void add(TOKEN_VARIANT x) { // add integer value to domain
            add_internal(x, (int)input_domain.size());
        }
        void add(TOKEN_VARIANT x, int y) {
            // ensure contiguous domain
            if (contiguous_output.find(y) == contiguous_output.end()) {
                int current_size = contiguous_output.size();
                contiguous_output.insert( std::make_pair(y,current_size) );
            }
            add_internal(x, contiguous_output[y]);
        }
        int encode(TOKEN_VARIANT x) {
            auto it = mapping.find(x);
            if (it == mapping.end()) {
                throw std::runtime_error("TOKEN VALUE IS OUT OF RANGE!");
            }
            return it->second;
        }
        int token_count; // number of individual tokens in the domain
        std::vector<midi::TOKEN_TYPE> repeat_tt; // repeat token types, idk what a repeat token type is 
        std::vector<std::tuple<TOKEN_VARIANT,int>> map_items;
        std::map<TOKEN_VARIANT,int> mapping; // same thing as map_items
        std::set<TOKEN_VARIANT> input_domain;
        std::set<int> output_domain;
        std::map<int,int> contiguous_output; // ensure contiguous output domain
        std::vector<TOKEN_INPUT_TYPE> input_types; // keep track of data types
    };
}
