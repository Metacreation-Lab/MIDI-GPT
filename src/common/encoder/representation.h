#pragma once

#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <string>
#include <iostream>
#include <sstream>
#include <variant>

#include "token_domain.h"
#include "../data_structures/verbosity.h"
#include "../midi_parsing/util_protobuf.h"

// START OF NAMESPACE
namespace encoder {

class REPRESENTATION {

/*
This class describes the token representation, ie. the vocabulary. It tracks the index and domain size (how many values per token type) 
and is used for one hot encoding and for sampling.
Therefore, it encodes a token from its base state (TOKEN_TYPE, value) to its vectorial one-hot encoded state. It also allows the reverse process.
*/

public:
  REPRESENTATION(std::vector<std::pair<midi::TOKEN_TYPE,TOKEN_DOMAIN>> spec) {

    /*
    params: spec - vector tht holds each token type and the allocated domain size in the vocabulary
    */

    // intialize vocabulary size
    vocab_size = 0;
    for (const auto &token_domain : spec) {
      // loop through each token type
      midi::TOKEN_TYPE tt = std::get<0>(token_domain); // token type
      TOKEN_DOMAIN domain = std::get<1>(token_domain); // token domain size

      int index = 0;
      for (const auto &value : domain.map_items) {
        // loop through each allocated token in the token domain
        int token = vocab_size + std::get<1>(value);
        TOKEN_TUPLE toktup = std::make_tuple(tt,std::get<0>(value));

        if (domain.repeat_tt.size() == 1) {
          token = forward[std::make_tuple(domain.repeat_tt[0],std::get<0>(value))];
        }

        forward[toktup] = token;
        if (domain.input_types[index] != TI_INT) {
          forward[std::make_tuple(tt,std::get<1>(value))] = token;
        }

        if (domain.repeat_tt.size() == 0) {
          backward[token] = toktup;
          backward_types[token] = domain.input_types[index];
        }
        index++;
      }
      vocab_size += domain.output_domain.size(); // add the current domain size to the vocabulary size
      domains.insert( std::make_pair(tt,domain.output_domain.size()) );
      token_domains.insert( std::make_pair(tt,domain) );
    }
  }
  int encode(midi::TOKEN_TYPE tt, TOKEN_VARIANT value) {
    std::tuple<midi::TOKEN_TYPE,TOKEN_VARIANT> key = std::make_tuple(tt,value);
    auto it = forward.find(key);
    if (it == forward.end()) {
      std::ostringstream buffer;
      auto tdit = token_domains.find(tt);
      if (tdit == token_domains.end()) {
        buffer << "ENCODER ERROR : TOKEN TYPE " << util_protobuf::enum_to_string(tt) << " IS NOT IN REPRESENTATION";
      }
      else {
        TOKEN_INPUT_TYPE ti = tdit->second.input_types[0];
        buffer << "ENCODER ERROR : VALUE (" << token_variant_to_string(ti, value) << ") NOT IN DOMAIN FOR TOKEN TYPE " << util_protobuf::enum_to_string(tt);
      }
      throw std::runtime_error(buffer.str());
    }
    return it->second;
  }
  int encode_partial(midi::TOKEN_TYPE tt, TOKEN_VARIANT value) {
    auto it = token_domains.find(tt);
    if (it == token_domains.end()) {
      throw std::runtime_error("midi::TOKEN_TYPE NOT PART OF THIS REPRESENTATION");
    }
    return it->second.encode(value);
  }
  int encode_partial_py_int(midi::TOKEN_TYPE tt, int value) {
    auto it = token_domains.find(tt);
    if (it == token_domains.end()) {
      throw std::runtime_error("midi::TOKEN_TYPE NOT PART OF THIS REPRESENTATION");
    }
    return it->second.encode(value);
  }
  void token_in_range(int token) {
    if (token >= vocab_size) {
      throw std::runtime_error("TOKEN IS LARGER THAN VOCAB SIZE!");
    }
    if (token < 0) {
      throw std::runtime_error("TOKEN IS NEGATIVE!");
    }
  }
  int decode(int token) {
    token_in_range(token);
    if (backward_types[token] != TI_INT) {
      throw std::runtime_error("TOKEN CAN NOT BE DECODED AS INT");
    }
    return std::get<int>(std::get<1>(backward[token]));
  }
  std::string decode_string(int token) {
    token_in_range(token);
    if (backward_types[token] != TI_STRING) {
      throw std::runtime_error("TOKEN CAN NOT BE DECODED AS STRING");
    }
    return std::get<std::string>(std::get<1>(backward[token]));
  }
  std::tuple<int,int> decode_timesig(int token) {
    token_in_range(token);
    if (backward_types[token] != TI_TIMESIG) {
      throw std::runtime_error("TOKEN CAN NOT BE DECODED AS TIMESIG");
    }
    return std::get<std::tuple<int,int>>(std::get<1>(backward[token]));
  }
  int max_token() {
    return vocab_size;
  }
  int get_domain_size(midi::TOKEN_TYPE tt) {
    auto it = domains.find(tt);
    if (it == domains.end()) {
      return 0;
    }
    return it->second;
  }
  bool in_domain(midi::TOKEN_TYPE tt, int value) {
    auto it = token_domains.find(tt);
    if (it != token_domains.end()) {
      return it->second.output_domain.find(value) != it->second.output_domain.end();
    }
    return false;
  }

  std::vector<int> get_num_bars_domain() {
    std::vector<int> model_dims;
    auto itt = token_domains.find(midi::TOKEN_NUM_BARS);
    if (itt != token_domains.end()) {
      for (const auto &value : itt->second.input_domain) {
        model_dims.push_back( std::get<int>(value) );
      }
    }
    return model_dims;
  }
  std::vector<std::tuple<int,int>> get_time_signature_domain() {
    std::vector<std::tuple<int,int>> timesigs;
    auto itt = token_domains.find(midi::TOKEN_TIME_SIGNATURE);
    if (itt != token_domains.end()) {
      for (const auto &ts : itt->second.input_domain) {
        timesigs.push_back( std::get<std::tuple<int,int>>(ts) );
      }
    }
    else {
      // the standard models without time signatures only trained on 4/4
      timesigs.push_back( std::make_tuple(4,4) );
    }
    return timesigs;
  }

  void check_token(int token) {
    auto it = backward.find(token);
    if (it == backward.end()) {
      std::ostringstream buffer;
      buffer << "ENCODER ERROR : TOKEN " << token << "IS NOT IN REPRESENTATION";
      throw std::runtime_error(buffer.str());
    }
  }
  bool is_token_type(int token, midi::TOKEN_TYPE tt) {
    check_token(token);
    return std::get<0>(backward[token]) == tt;
  }
  midi::TOKEN_TYPE get_token_type(int token) {
    check_token(token);
    return std::get<0>(backward[token]);
  }
  bool has_token_type(midi::TOKEN_TYPE tt) {
    return token_domains.find(tt) != token_domains.end();
  }
  bool has_token_types(std::vector<midi::TOKEN_TYPE> tts) {
    for (const auto &tt : tts) {
      if (!has_token_type(tt)) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  std::vector<T> get_mask(T value) {
    return std::vector<T>(vocab_size, value);
  }

  template <typename T>
  std::set<midi::TOKEN_TYPE> get_mask_token_types(std::vector<T> &mask) {
    std::set<midi::TOKEN_TYPE> tts;
    for (int i=0; i<(int)mask.size(); i++) {
      if (mask[i] > 0) {
        tts.insert( get_token_type(i) );
      }
    }
    return tts;
  }

  template <typename T>
  void show_mask_token_types(std::vector<T> &mask) {
    std::set<midi::TOKEN_TYPE> tts = get_mask_token_types(mask);
    data_structures::LOGGER("MASK TOKEN TYPES :: ");
    for (const auto &tt : tts) {
      data_structures::LOGGER(data_structures::to_str(util_protobuf::enum_to_string(tt), ", "), false);
    }
    data_structures::LOGGER("");
  }

  template <typename T>
  void set_mask(midi::TOKEN_TYPE tt, std::vector<int> values, std::vector<T> &mask, T mask_value) {
    auto it = token_domains.find(tt);
    if (it != token_domains.end()) {
      for (const auto &value : values) {
        if (value == -1) {
          for (const auto &v : it->second.input_domain) {
            mask[encode(tt, v)] = mask_value;
          }
        }
        else {
          mask[encode(tt, value)] = mask_value;
        }
      }
    }
  }

  template <typename T>
  void set_mask(midi::TOKEN_TYPE tt, std::vector<std::string> values, std::vector<T> &mask, T mask_value, STRING_VECTOR_FLAG x) {
    auto it = token_domains.find(tt);
    if (it != token_domains.end()) {
      for (const auto &value : values) {
        mask[encode(tt, value)] = mask_value;
      }
    }
  }

  std::vector<int> encode_to_one_hot(midi::TOKEN_TYPE tt, std::vector<int> values) {
    std::vector<int> x(vocab_size,0);
    set_mask(tt, values, x, 1);
    return x;
  }

  std::vector<int> get_type_mask(std::vector<midi::TOKEN_TYPE> tts) {
    std::vector<int> mask(vocab_size,0);
    for (int i=0; i<vocab_size; i++) {
      for (const auto &tt : tts) {
        if (is_token_type(i,tt)) {
          mask[i] = 1;
          break;
        }
      }
    }
    return mask;
  }

  std::string token_variant_to_string(TOKEN_INPUT_TYPE ti, TOKEN_VARIANT v) {
    std::string value_str;
    if (ti == TI_INT) {
      value_str = std::to_string(std::get<int>(v));
    }
    else if (ti == TI_STRING) {
      value_str = std::get<std::string>(v);
    }
    else if (ti == TI_TIMESIG) {
      auto ts = std::get<std::tuple<int,int>>(v);
      value_str = std::to_string(std::get<0>(ts)) + "/" + std::to_string(std::get<1>(ts));
    }
    else {
      throw std::runtime_error("THE TOKEN HAS NO INVALID TOKEN_INPUT_TYPE");
    }
    return value_str;
  }

  std::string pretty(int token) {
    auto token_value = backward[token];
    TOKEN_INPUT_TYPE ti = backward_types[token];
    return util_protobuf::enum_to_string(std::get<0>(token_value)) + std::string(" = ") + token_variant_to_string(ti, std::get<1>(token_value));
  }

  std::string pretty_type(int token) {
    auto token_value = backward[token];
    return util_protobuf::enum_to_string(std::get<0>(token_value));
  }

  void show(std::vector<int> &tokens) {
    for (const auto &token : tokens) {
      data_structures::LOGGER(pretty(token));
    }
  }

  void show_token_types() {
    for (const auto &token : domains) {
      data_structures::LOGGER(data_structures::to_str("REP TOKENS :: ", util_protobuf::enum_to_string(token.first)));
    }
  }

  void show_mapping() {
    for (int i=0; i<vocab_size; i++) {
      std::cout << i << "\t" << pretty(i) << std::endl;
    }
  }

  // function to determine if pretrain instrument mapping is used
  bool has_pretrain_instrument_mapping() {
    auto it = token_domains.find(midi::TOKEN_INSTRUMENT);
    if (it != token_domains.end()) {
      return (it->second.output_domain.size() < 128);
    }
    return false;
  }

  int vocab_size;
  std::map<TOKEN_TUPLE,int> forward;
  std::map<int,TOKEN_TUPLE> backward;
  std::map<int,TOKEN_INPUT_TYPE> backward_types;

  std::map<midi::TOKEN_TYPE,int> domains; // maps each token type to its domain output size
  std::map<midi::TOKEN_TYPE,TOKEN_DOMAIN> token_domains; // maps each token type to its token domain
};

}
// END OF NAMESPACE
