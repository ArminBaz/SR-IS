# Adding packages

] activate RL
] add blah

import RL

# Notes

There's some ambiguity here between handling Q-values and V.

The code is effectively using V as a stand-in for Q, which is fine so long as rewards are only in terminal states.

The only exception is MB: value iteration requires us to keep them separate.

Note that rather than a pure stand-in, we /do/ discount V by gamma to obtain Q. The reason for this is
consistency with the softmax choice rule, which is operating on discounted values.

The big advantage to sticking with Q-values is a much simpler and more intutive SR M matrix, where M is simply
state-to-state future state occupancy.