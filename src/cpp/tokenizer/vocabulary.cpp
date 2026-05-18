#include "vocabulary.h"
#include <stdexcept>

namespace midigpt::tokenizer {

Vocabulary::Vocabulary(const EncoderConfig& config) : config_(config) {
    vocab_size_ = 0;
    for (const auto& td : config_.token_domains) {
        offsets_[td.type] = vocab_size_;
        vocab_size_ += td.domain_size;
    }
}

int Vocabulary::encode(TokenType type, int value) const {
    auto it = offsets_.find(type);
    if (it == offsets_.end()) {
        throw std::runtime_error("TokenType not found in vocabulary");
    }
    if (value < 0 || value >= domain_size(type)) {
        throw std::runtime_error("Token value out of domain size");
    }
    return it->second + value;
}

std::pair<TokenType, int> Vocabulary::decode(int token) const {
    if (token < 0 || token >= vocab_size_) {
        throw std::runtime_error("Token out of vocabulary range");
    }
    for (const auto& td : config_.token_domains) {
        int start = offsets_.at(td.type);
        if (token >= start && token < start + td.domain_size) {
            return {td.type, token - start};
        }
    }
    throw std::runtime_error("Token decoding failed");
}

int Vocabulary::size() const {
    return vocab_size_;
}

bool Vocabulary::has(TokenType type) const {
    return offsets_.find(type) != offsets_.end();
}

int Vocabulary::domain_size(TokenType type) const {
    for (const auto& td : config_.token_domains) {
        if (td.type == type) return td.domain_size;
    }
    return 0;
}

int Vocabulary::offset(TokenType type) const {
    auto it = offsets_.find(type);
    if (it != offsets_.end()) return it->second;
    return -1;
}

bool Vocabulary::is_type(int token, TokenType type) const {
    int o = offset(type);
    if (o == -1) return false;
    return token >= o && token < o + domain_size(type);
}

TokenType Vocabulary::get_type(int token) const {
    return decode(token).first;
}

std::pair<int,int> Vocabulary::range(TokenType type) const {
    int o = offset(type);
    if (o == -1) return {-1, -1};
    return {o, o + domain_size(type)};
}

} // namespace midigpt::tokenizer
