#include "agent_state.h"

#include <gtest/gtest.h>

using pperf::AgentState;
using pperf::InvalidTransition;
using pperf::StateMachine;

TEST(AgentState, AcceptsCaptureAndRepeatedReplayCycle) {
  StateMachine state;
  state.transition(AgentState::warming);
  state.transition(AgentState::ready);
  state.transition(AgentState::armed);
  state.transition(AgentState::capturing);
  state.transition(AgentState::quiescing);
  state.transition(AgentState::frozen);
  for (int index = 0; index < 3; ++index) {
    state.transition(AgentState::replay_prepared);
    state.transition(AgentState::replaying);
    state.transition(AgentState::frozen);
  }
  state.transition(AgentState::shutdown);
  EXPECT_EQ(state.state(), AgentState::shutdown);
}

TEST(AgentState, RejectsSkippedAndPostShutdownTransitions) {
  StateMachine state;
  EXPECT_THROW(state.transition(AgentState::ready), InvalidTransition);
  state.transition(AgentState::warming);
  state.transition(AgentState::shutdown);
  EXPECT_THROW(state.transition(AgentState::warming), InvalidTransition);
}
