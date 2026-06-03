from .filter_acceptance import Filter
from .filter_line_search import FilterLineSearch, LineSearch


def make_line_search(options, problem, optimizer):
    if isinstance(options["line_search"], LineSearch):
        return options["line_search"]
    elif options["line_search"] == "filter":
        return FilterLineSearch(options, problem, optimizer)
    else:
        search = options["line_search"]
        raise ValueError(f"Unrecognized line_search {search}")
