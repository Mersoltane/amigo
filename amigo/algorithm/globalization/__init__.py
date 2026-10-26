"""Globalization: line searches (filter, funnel, merit) and the
feasibility restoration phase they fall back to."""

from .feasibility_restoration import FeasibilityRestoration
from .filter_acceptance import Filter
from .filter_line_search import FilterLineSearch, LineSearch
from .funnel_line_search import FunnelLineSearch
from .merit_line_search import MeritLineSearch


def make_restoration(options, problem, optimizer):
    return FeasibilityRestoration(options, problem, optimizer)


def make_line_search(options, problem, optimizer):
    if isinstance(options["line_search"], LineSearch):
        return options["line_search"]
    elif options["line_search"] == "filter":
        return FilterLineSearch(options, problem, optimizer)
    elif options["line_search"] == "funnel":
        return FunnelLineSearch(options, problem, optimizer)
    elif options["line_search"] == "merit":
        return MeritLineSearch(options, problem, optimizer)
    else:
        search = options["line_search"]
        raise ValueError(f"Unrecognized line_search {search}")
