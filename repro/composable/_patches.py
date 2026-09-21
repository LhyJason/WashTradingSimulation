r"""
_patches.py -- runtime patches (monkey-patches); does NOT modify any source file in marketsim/.

These are applied on import of repro.composable. The on-disk marketsim/ files stay untouched;
to disable, remove the import of this module from __init__.py.

------------------------------------------------------------------------------------------
Patch 1 -- None-safe Order.__eq__
------------------------------------------------------------------------------------------
The upstream marketsim/fourheap/order.py defines Order.__eq__ as
        return self.order_id == other.order_id
with no handling for `other` being None / a non-Order. So any `some_order != None`
(e.g. hbl_agent.py:575 checking whether the book is non-empty) triggers None.order_id and
raises AttributeError. This is a real bug on the upstream HBL path; pure-ZI experiments
never reach that branch, so it stayed hidden. We replace __eq__ with a None-safe but
otherwise equivalent version (compares False against None / non-Order, else by order_id).

------------------------------------------------------------------------------------------
Patch 2 -- dense LazyGaussianMeanReverting.get_value_at (so InformedZI can be mixed)
------------------------------------------------------------------------------------------
The upstream lazy fundamental only stores the value at *queried* times and jumps forward
(its _generate_at requires monotonically increasing query times). InformedZI calls
get_final_fundamental() on every action, which pushes latest_t to T; afterwards any agent
that reads the *current* fundamental at t < T hits dt = t - T < 0 and crashes
(torch.randn(negative)). We replace get_value_at with a version that fills EVERY integer
step up to the requested time and stores each one, so later queries for earlier times are
plain dict lookups -- arbitrary query order becomes safe. This is the same AR(1) process
(paper Eq. (1)); the realized path differs from the upstream lazy version (random draws are
consumed differently) but the distribution is identical.
"""
import torch

from marketsim.fourheap.order import Order
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting

# --- Patch 1: None-safe Order.__eq__ (idempotency guard avoids patching twice) ---
if not getattr(Order, "_eq_none_safe_patched", False):
    _original_eq = Order.__eq__

    def _none_safe_eq(self, other):
        if other is None or not isinstance(other, Order):
            return False
        return self.order_id == other.order_id

    Order.__eq__ = _none_safe_eq
    Order._eq_none_safe_patched = True


# --- Patch 2: dense fundamental generation (arbitrary query order safe) ---
if not getattr(LazyGaussianMeanReverting, "_dense_fill_patched", False):

    def _get_value_at_dense(self, time):
        """Fill every integer step up to `time` (storing each); AR(1): f_t = r*mean + (1-r)*f_{t-1} + eps."""
        time = int(time)
        if self.latest_t < time:
            n = time - self.latest_t
            shocks = torch.randn(n) * self.shock_std + self.shock_mean
            r = self.r.item()
            mean = self.mean.item()
            prev = self.fundamental_values[self.latest_t]
            for i in range(n):
                prev = r * mean + (1 - r) * prev + shocks[i].item()
                self.latest_t += 1
                self.fundamental_values[self.latest_t] = prev
        return self.fundamental_values[time]

    LazyGaussianMeanReverting.get_value_at = _get_value_at_dense
    LazyGaussianMeanReverting._dense_fill_patched = True
