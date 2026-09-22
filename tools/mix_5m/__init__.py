"""Scaling authorisation for `decision-mix-clean-5m` (#T-mix-5m, data-training §§89, 136).

There is no mixer here, on purpose. #T-mix-5m's verification gate makes the
5 M corpus CONDITIONAL — «si 1 M no mejora, esta task registra NO-GO y no
genera 5 M» — so the only thing this package owns is the decision and the
rule that keeps the published decision honest.
"""
