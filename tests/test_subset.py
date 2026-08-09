from pmesa.evidence import EvidenceKind, EvidenceUnit
from pmesa.subset import SubsetObjective, greedy_select


def test_greedy_selection_is_reproducible_and_uses_diversity():
    units = [
        EvidenceUnit("a", EvidenceKind.VISUAL, "a", {"object": 1}),
        EvidenceUnit("b", EvidenceKind.VISUAL, "b", {"object": 1}),
        EvidenceUnit("c", EvidenceKind.TEXTUAL, "c", {"attribute": 1}),
    ]
    objective = SubsetObjective(
        units, [1, .99, .98], [1, .99, .98], [1, 1, 1], normalize=False
    )
    selected = greedy_select(objective, 2)
    assert selected == greedy_select(objective, 2)
    assert 2 in selected


def test_objective_is_monotone_for_nonnegative_inputs():
    units = [EvidenceUnit(str(i), EvidenceKind.VISUAL, str(i), {str(i): 1}) for i in range(4)]
    objective = SubsetObjective(units, [1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1])
    assert objective([0, 1, 2]) >= objective([0, 1]) >= objective([0]) >= objective([])
