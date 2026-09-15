#include "agent_state.h"

#include <sstream>

namespace pperf {

const char* state_name(AgentState state) noexcept {
  switch (state) {
    case AgentState::initializing: return "initializing";
    case AgentState::warming: return "warming";
    case AgentState::ready: return "ready";
    case AgentState::armed: return "armed";
    case AgentState::capturing: return "capturing";
    case AgentState::quiescing: return "quiescing";
    case AgentState::frozen: return "frozen";
    case AgentState::replay_prepared: return "replay_prepared";
    case AgentState::replaying: return "replaying";
    case AgentState::shutdown: return "shutdown";
  }
  return "invalid";
}

void StateMachine::transition(AgentState next) {
  bool valid = false;
  switch (state_) {
    case AgentState::initializing:
      valid = next == AgentState::warming;
      break;
    case AgentState::warming:
      valid = next == AgentState::ready || next == AgentState::shutdown;
      break;
    case AgentState::ready:
      valid = next == AgentState::armed || next == AgentState::shutdown;
      break;
    case AgentState::armed:
      valid = next == AgentState::capturing || next == AgentState::shutdown;
      break;
    case AgentState::capturing:
      valid = next == AgentState::quiescing || next == AgentState::shutdown;
      break;
    case AgentState::quiescing:
      valid = next == AgentState::frozen || next == AgentState::shutdown;
      break;
    case AgentState::frozen:
      valid = next == AgentState::replay_prepared ||
              next == AgentState::shutdown;
      break;
    case AgentState::replay_prepared:
      valid = next == AgentState::replaying || next == AgentState::shutdown;
      break;
    case AgentState::replaying:
      valid = next == AgentState::frozen || next == AgentState::shutdown;
      break;
    case AgentState::shutdown:
      valid = false;
      break;
  }
  if (!valid) {
    std::ostringstream message;
    message << "capsule_invalid: invalid agent transition "
            << state_name(state_) << " -> " << state_name(next);
    throw InvalidTransition(message.str());
  }
  state_ = next;
}

void StateMachine::require(AgentState expected, const char* operation) const {
  if (state_ == expected) return;
  std::ostringstream message;
  message << "capsule_invalid: " << operation << " requires "
          << state_name(expected) << ", observed " << state_name(state_);
  throw InvalidTransition(message.str());
}

}  // namespace pperf
