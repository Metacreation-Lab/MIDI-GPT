#pragma once

#include <chrono>
#include <ctime>

namespace sampling {

  // Base class for callbacks
  class CallbackBase {
  public:
    CallbackBase () { }
    virtual ~CallbackBase () { }
    virtual void on_bar_end () {}
    virtual void on_prediction (std::vector<float> &logits, int next_token) {}
    virtual void on_start () {}
    virtual float update_temperature(float current_temperature) {
      return current_temperature;
    }
    virtual bool is_cancelled() {
      return false;
    }
  };

  // Class that manages call all callbacks
  class CallbackManager {
  public:
    CallbackManager () {}
    ~CallbackManager () {}
    void add_callback_ptr(std::shared_ptr<CallbackBase> x) {
      callbacks.push_back(x);
    }
    void on_bar_end () {
      for (auto &x : callbacks) {
        x->on_bar_end();
      }
    }
    void on_prediction (std::vector<float> &logits, int next_token) {
      for (auto &x : callbacks) {
        x->on_prediction(logits, next_token);
      }
    }
    void on_start () {
      for (auto &x : callbacks) {
        x->on_start();
      }
    }
    float update_temperature (float current_temperature) {
      for (auto &x : callbacks) {
        float value = x->update_temperature(current_temperature);
        if (value > current_temperature) {
          return value;
        }
      }
      return current_temperature;
    }
    bool is_cancelled() {
      for (auto &x : callbacks) {
        if (x->is_cancelled()) {
          return true;
        }
      }
      return false;
    }
    std::vector<std::shared_ptr<CallbackBase>> callbacks;
  };


  // Callback examples
  class TemperatureIncreaseCallback : public CallbackBase {
  public:
    TemperatureIncreaseCallback (float _increase, float _current_temperature) {
      increase = _increase;
      current_temperature = _current_temperature;
    }
    float update_temperature(float temp) {
      current_temperature = temp + increase;
      std::cout << "CURRENT TEMPERATURE : " << current_temperature << std::endl;
      return current_temperature;
    }
    float increase;
    float current_temperature;
  };


  class LogLikelihoodCallback : public CallbackBase {
  public:
    LogLikelihoodCallback () {
      loglik = 0;
      sequence_length = 0;
    }
    void on_prediction(std::vector<float> &logits, int next_token) {
      loglik += logits[next_token];
      sequence_length++;
    }
    void on_start() {
      loglik = 0;
      sequence_length = 0;
    }
    double loglik;
    int sequence_length;
  };

  class RecordTokenSequenceCallback : public CallbackBase {
  public:
    RecordTokenSequenceCallback () {}
    void on_start() {
      tokens.clear();
    }
    void on_prediction(std::vector<float> &logits, int next_token) {
      tokens.push_back(next_token);
    }
    std::vector<int> tokens;
  };

  class CancelCallback : public CallbackBase {
  public:
    CancelCallback () {
      cancel = false;
    }
    void set_cancel(bool cancel_value) {
      cancel = cancel_value;
    }
    bool is_cancelled() {
      return cancel;
    }
    bool cancel;
  };

}