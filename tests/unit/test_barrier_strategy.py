"""
Unit tests for the quality-function barrier strategy's globalization
logic: the obj-constr-filter progress check and its point memory.
Pure-logic tests via __new__, no compiled backend required.
"""

from amigo.algorithm.barrier_strategy.quality_function import (
    QualityFunctionBarrierStrategy,
)


class StubEvaluator:
    def evaluate_objective_and_infeasibility(self, state):
        pass


class StubState:
    def __init__(self, f, theta):
        self.objective_value = f
        self.log_barrier_value = 0.0
        self.con_infeasibility = theta

    @property
    def barrier_objective(self):
        return self.objective_value + self.log_barrier_value


def make_strategy(**options):
    qf = QualityFunctionBarrierStrategy.__new__(QualityFunctionBarrierStrategy)
    qf.options = {
        "adaptive_mu_globalization": "obj-constr-filter",
        "filter_margin_fact": 1e-5,
        "filter_max_margin": 1.0,
        **options,
    }
    qf.glob_filter = []
    qf.refs = []
    return qf


class TestObjConstrFilterGlobalization:
    def test_empty_filter_is_sufficient(self):
        qf = make_strategy()
        assert qf._sufficient_progress(StubEvaluator(), StubState(1.0, 1.0))

    def test_progress_requires_beating_every_entry(self):
        """A point dominated by any entry is insufficient, even if it
        improves on some other, staler entry."""
        qf = make_strategy()
        ev = StubEvaluator()
        qf._remember_point(ev, StubState(10.0, 10.0))
        qf._remember_point(ev, StubState(1.0, 1.0))
        # Beats the stale (10, 10) entry but is dominated by (1, 1)
        assert not qf._sufficient_progress(ev, StubState(5.0, 5.0))

    def test_improvement_in_one_coordinate_is_progress(self):
        qf = make_strategy()
        ev = StubEvaluator()
        qf._remember_point(ev, StubState(1.0, 1.0))
        assert qf._sufficient_progress(ev, StubState(2.0, 0.5))
        assert qf._sufficient_progress(ev, StubState(0.5, 2.0))

    def test_stagnation_is_not_progress(self):
        qf = make_strategy()
        ev = StubEvaluator()
        qf._remember_point(ev, StubState(1.0, 1.0))
        assert not qf._sufficient_progress(ev, StubState(1.0, 1.0))

    def test_margin_rejects_marginal_progress(self):
        qf = make_strategy(filter_margin_fact=1e-2)
        ev = StubEvaluator()
        qf._remember_point(ev, StubState(1.0, 1.0))
        # Better than (1, 1) but within the 1e-2 margin in both coords
        assert not qf._sufficient_progress(ev, StubState(1.0 - 1e-3, 1.0 - 1e-3))

    def test_feasible_entries_demand_objective_decrease(self):
        """At theta = 0 the margin vanishes and progress means a strict
        objective decrease."""
        qf = make_strategy()
        ev = StubEvaluator()
        qf._remember_point(ev, StubState(1.0, 0.0))
        assert not qf._sufficient_progress(ev, StubState(1.0, 0.0))
        assert qf._sufficient_progress(ev, StubState(0.9, 0.0))

    def test_margin_scale_is_capped(self):
        """Large violations use the max_margin cap, not theta itself."""
        qf = make_strategy(filter_margin_fact=0.1, filter_max_margin=1.0)
        ev = StubEvaluator()
        qf._remember_point(ev, StubState(100.0, 50.0))
        # Entry is (100 - 0.1, 50 - 0.1), not (100 - 5, 50 - 5)
        assert not qf._sufficient_progress(ev, StubState(99.95, 49.95))
        assert qf._sufficient_progress(ev, StubState(99.85, 49.95))

    def test_never_monotone_is_always_sufficient(self):
        qf = make_strategy(adaptive_mu_globalization="never-monotone")
        assert qf._sufficient_progress(StubEvaluator(), StubState(1.0, 1.0))


class TestModeSwitching:
    def make_monotone(self):
        qf = make_strategy(verbose_barrier=False)
        qf.free_mode = False
        qf._mu_min_default = False
        qf.mu_min = 1e-11
        qf.mu_max = 1e3
        qf.init_dual_inf = 1.0
        return qf

    def test_progress_in_monotone_mode_returns_to_free_mode(self):
        """The point that restores free mode is not retested against itself."""
        qf = self.make_monotone()
        ev = StubEvaluator()
        qf._remember_point(ev, StubState(10.0, 10.0))

        state = StubState(1.0, 1.0)
        state.mu = 0.1
        state.comm_rank = 0
        info = qf.update_barrier(ev, state)

        assert qf.free_mode
        assert info.new_barrier
        assert len(qf.glob_filter) == 2

    def test_backtracking_keeps_free_mode(self):
        qf = make_strategy()
        qf.free_mode = True
        info = type("Info", (), {"success": True, "num_search_iters": 4})()
        qf.update_after_line_search(info, StubEvaluator(), StubState(1.0, 1.0))
        assert qf.free_mode
