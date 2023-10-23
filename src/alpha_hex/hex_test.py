#!/usr/bin/env python3
import hex as hx

game = hx.Hex()
state = game._initial_state()

state = game._next_state(state, 0)
state = game._next_state(state, 1)
state = game._next_state(state, 2)
state = game._next_state(state, 3)
state = game._next_state(state, 4)
encoded_state = game.get_encoded_state(state)

print(state._board)
print(encoded_state)
