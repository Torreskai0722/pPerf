#pragma once

#include <stdexcept>
#include <string>

namespace pperf {

enum class AgentState {
  initializing,
  warming,
  ready,
  armed,
  capturing,
  quiescing,
  frozen,
  replay_prepared,
  replaying,
  shutdown,
};

const char* state_name(AgentState state) noexcept;

class InvalidTransition : public std::runtime_error {
 public:
  explicit InvalidTransition(const std::string& message)
      : std::runtime_error(message) {}
};

class StateMachine {
 public:
  AgentState state() const noexcept { return state_; }
  void transition(AgentState next);
  void require(AgentState expected, const char* operation) const;

 private:
  AgentState state_{AgentState::initializing};
};

}  // namespace pperf
