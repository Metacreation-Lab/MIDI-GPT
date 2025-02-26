#pragma once

#include <vector>
#include "midi.pb.h"
#include "../encoder/representation.h"

namespace data_structures {

    class TokenSequence {
        public:
        TokenSequence(const std::shared_ptr<encoder::REPRESENTATION> &rep) {
            bar_num = 0;
            track_num = 0;
        }
        void push_back( int token ) {
            tokens.push_back( token );
        }

        void insert( std::vector<int> &tokens) {
            for (auto token : tokens) {
                push_back(token);
            }
        }

        void on_track_start(midi::Piece *x, const std::shared_ptr<encoder::REPRESENTATION> &rep) {
            track_num++;
            bar_num = 0;
        }
        void on_bar_start(midi::Piece *x, const std::shared_ptr<encoder::REPRESENTATION> &rep) {

            bar_num++;
        }
        int bar_num;
        int track_num;
        
        std::vector<int> tokens;
    };
}
