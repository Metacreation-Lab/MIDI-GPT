#pragma once

#include "encoder_config.h"
#include <unordered_map>
#include <tuple>
#include <stdexcept>

namespace midigpt::tokenizer {

class Vocabulary {
public:
    explicit Vocabulary(const EncoderConfig& config);

    // Encoding
    int encode(TokenType type, int value)               const;

    // Decoding
    std::pair<TokenType, int> decode(int token)         const;

    // Queries
    int  size()                                         const;
    bool has(TokenType type)                            const;
    int  domain_size(TokenType type)                    const;
    int  offset(TokenType type)                         const; // start index in vocab
    bool is_type(int token, TokenType type)             const;
    TokenType get_type(int token)                       const;

    // Range [start, end) of token integers for a given type — used by masking
    std::pair<int,int> range(TokenType type)            const;

    const EncoderConfig& config() const { return config_; }

private:
    EncoderConfig                     config_;
    std::unordered_map<TokenType,int> offsets_;
    int                               vocab_size_;
};

} // namespace midigpt::tokenizer
