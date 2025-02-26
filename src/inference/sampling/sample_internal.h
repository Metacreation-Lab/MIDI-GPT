#pragma once

#include <ATen/core/ivalue.h>
#include <torch/script.h>
#include <torch/nn/functional/activation.h>

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <set>

#include "../enum/model_type.h"
#include "../../common/data_structures/verbosity.h"
#include "control.h"
#include "callback_base.h"

namespace sampling {

  class ModelMeta {
  public:
    torch::jit::Module model;
    midi::ModelMetadata meta;
  };

  static const int NUM_LAYERS = 6;

  void load_checkpoint(const std::string &ckpt_path, const std::unique_ptr<ModelMeta> &m) { 
    try {
      std::unordered_map<std::string, std::string> loaded_extra_files;
      loaded_extra_files["metadata.json"] = "";
      m->model = torch::jit::load(ckpt_path, torch::kCPU, loaded_extra_files);
      if (loaded_extra_files["metadata.json"].size() == 0) {
        throw std::runtime_error("ERROR LOADING MODEL : MODEL CONTAINS NO METADATA!");
      }
      util_protobuf::string_to_protobuf(loaded_extra_files["metadata.json"], &m->meta);
      data_structures::LOGGER( "MODEL METADATA :" );
    }
    catch (const c10::Error& e) {
      data_structures::LOGGER( e.what() );
      throw std::runtime_error("ERROR LOADING MODEL.");
    }
  }

  std::unique_ptr<ModelMeta> load_model(midi::HyperParam *param) {
    auto model = std::make_unique<ModelMeta>();
    load_checkpoint(param->ckpt(), model);
    if (model->meta.model_dim() != -1) {
      param->set_model_dim(model->meta.model_dim());
    }

    model->meta.set_num_heads(8);
    model->meta.set_num_layers(6);

    return model;
  }

  void sample_inner(std::vector<std::unique_ptr<SAMPLE_CONTROL>> &scon, std::vector<std::vector<int>> &seqs, torch::jit::Module *model, std::vector<torch::jit::IValue> &inputs, midi::HyperParam *param, CallbackManager *callbacks) {

    if (!model) {
      throw std::runtime_error("ERROR : MODEL IS INVALID.");
    }

    torch::Tensor logits;
    torch::jit::IValue past_key_values;

    auto outputs = model->forward(inputs).toTuple();
    logits = outputs->elements()[0].toTensor().index(
      {torch::indexing::Slice(),-1,torch::indexing::Slice()});
    past_key_values = outputs->elements()[1];


    // get logits for first in batch
    std::vector<std::vector<int>> masks_copy;
    std::vector<std::vector<float>> logits_copy;
    for (int i=0; i<(int)seqs.size(); i++) {
      logits_copy.push_back(std::vector<float>(logits[i].data_ptr<float>(), logits[i].data_ptr<float>() + logits[i].numel()));
    }

    // set masks
    std::vector<std::set<midi::TOKEN_TYPE>> masked_tts;
    int num_masked = 0;
    for (int i=0; i<(int)seqs.size(); i++) {
      std::vector<std::string> unmasked_types;
      std::vector<int> mask = scon[i]->get_mask( seqs[i] );
      masks_copy.push_back( mask );
      masked_tts.push_back( scon[i]->rep->get_mask_token_types(mask) );
      scon[i]->rep->show_mask_token_types(mask);
      if ((!scon[i]->finished) && (!param->internal_disable_masking())) {
        for (int j=0; j<(int)mask.size(); j++) {
          if (mask[j] == 0) {
            logits[i][j] = -1 * std::numeric_limits<float>::max(); // set this to a very small possibility
            num_masked++;
          } else {
            unmasked_types.push_back(scon[i]->enc->rep->pretty_type(j));
          }
        }
      }
      std::set<std::string> s( unmasked_types.begin(), unmasked_types.end() );
      unmasked_types.assign( s.begin(), s.end() );
      for (auto strr : unmasked_types) {
        std::cout << "NOT MASKED: " << strr << std::endl;
      }

      if (param->mask_top_k() > 0) {

        std::mt19937 engine(time(NULL));

        // optionally mask the top k tokens
        bool can_mask = false;
        std::vector<midi::TOKEN_TYPE> token_types_to_mask = {midi::TOKEN_NOTE_ONSET, midi::TOKEN_TIME_ABSOLUTE_POS, midi::TOKEN_NOTE_DURATION};
        for (const auto &t : token_types_to_mask) {
          if (masked_tts[i].count(t) > 0) {
            can_mask = true;
            break;
          }
        }
        if ((can_mask) && (random_on_unit(&engine) < param->mask_top_k())) {
          std::vector<int> V(mask.size());
          std::iota(V.begin(),V.end(),0);
          std::sort( V.begin(),V.end(), [&](int ii,int jj){ return (logits[i][ii] > logits[i][jj]).item<int64_t>(); });

          for (int j=0; j<10; j++) {
            if (j==0) {
              logits[i][V[j]] = -1 * std::numeric_limits<float>::max();
              num_masked++;
            }
          }
        }
      }
    }

    if (param->sampling_seed() != -1) {
      torch::manual_seed(param->sampling_seed());
    }

    float temperature = param->temperature();
    auto probs = (logits / temperature).softmax(1);
    auto next_tokens = probs.multinomial(1);

    inputs.clear();
    inputs.push_back( next_tokens );
    inputs.push_back( past_key_values );
    
    // add next token to the sequences
    for (int i=0; i<(int)seqs.size(); i++) {
      if (!scon[i]->finished) {      
        int next_token = next_tokens[i][0].item<int64_t>();
        data_structures::LOGGER(data_structures::to_str("SAMPLED :: ", scon[i]->enc->rep->pretty(next_token)));
        seqs[i].push_back( next_token );


        if (callbacks) {
          if ((scon[i]->enc->rep->is_token_type(next_token, midi::TOKEN_BAR_END)) || (scon[i]->enc->rep->is_token_type(next_token, midi::TOKEN_FILL_IN_END))) {
            callbacks->on_bar_end();
          }
          callbacks->on_prediction(logits_copy[i], next_token);
        }
      }
    }
  }

  void make_state(std::vector<torch::jit::IValue> *state, int batch_size, midi::ModelMetadata *meta) {
      data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, "make_state" );
    for (int i=0; i<meta->num_layers(); i++) {
      std::vector<torch::jit::IValue> tuple;
      for (int j=0; j<2; j++) {
        tuple.push_back( torch::zeros({batch_size, meta->num_heads(), 0, meta->num_hidden()}) );
      }
      state->push_back( torch::ivalue::Tuple::create(tuple) );
    }
  }

  std::vector<midi::Piece> generate(midi::Status *status, midi::Piece *piece, midi::HyperParam *param, const std::unique_ptr<ModelMeta> &mm, CallbackManager *callbacks) {
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_DEBUG, "generate");
    data_structures::LOGGER(data_structures::VERBOSITY_LEVEL_TRACE, util_protobuf::protobuf_to_string(status));
    param->set_temperature( std::max((double)param->temperature(), 1e-6) ); // CAN'T HAVE ZERO TEMPERATURE
    std::vector<std::unique_ptr<SAMPLE_CONTROL>> scon;
    for (int i=0; i<param->batch_size(); i++) {
      scon.push_back( std::make_unique<SAMPLE_CONTROL>(piece, status, param, &mm->meta) );
    }
    for (auto &sc : scon) {
      data_structures::LOGGER("REG GRAPH" );
      sc->rg->graph.print_graphviz();
    }
    std::vector<int> prompt = scon[0]->prompt;
    std::vector<torch::jit::IValue> inputs;
    std::vector<std::vector<int>> seqs = std::vector<std::vector<int>>(param->batch_size(), prompt);
    scon[0]->rep->show(prompt);

    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor x = torch::zeros({param->batch_size(), (int)prompt.size()}, opts);
    for (int k=0; k<param->batch_size(); k++) {
      for (int i=0; i<(int)prompt.size(); i++) {
        x[k][i] = prompt[i];
      }
    }
    inputs.push_back( x );
    std::vector<torch::jit::IValue> state;
    if ((param) && (mm->meta.new_state())) {
        make_state(&state, param->batch_size(), &mm->meta);
    }
    inputs.push_back(torch::ivalue::Tuple::create(state));


    bool terminated = false;
    int num_steps = 0;
    while (!scon[0]->finished) {
      sample_inner(scon, seqs, &mm->model, inputs, param, callbacks);
      num_steps++;
      if ((param->max_steps() > 0) && (num_steps >= param->max_steps())) {
        terminated = true;
        break;
      }
      if ((callbacks) && (callbacks->is_cancelled())) {
        terminated = true;
        break;
      }
    }
    scon[0]->enc->config->decode_final = status->decode_final();
    scon[0]->rep->show(seqs[0]);
    std::vector<midi::Piece> output(param->batch_size());
    if (!terminated) {
      scon[0]->enc->tokens_to_json_array(seqs, output);
      scon[0]->finalize(&output[0]); // batch size should be 1 anyways
    }
    return output;
  }

}