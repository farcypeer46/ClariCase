from src.utils import labels as L
from src.utils.config import ISSUE_VARIANT_MERGES


def test_canonical_issue_applies_merges():
    for src, dst in ISSUE_VARIANT_MERGES.items():
        assert L.canonical_issue(src) == dst
    assert L.canonical_issue("T02_I001") == "T02_I001"


def test_issue_to_team_and_reverse_are_consistent():
    for iid, tid in L.issue_to_team.items():
        assert iid in L.team_to_issues[tid]
    for tid, iids in L.team_to_issues.items():
        assert all(L.issue_to_team[i] == tid for i in iids)


def test_team_issue_mask_shape_and_diagonal():
    team_classes = sorted(L.team_to_issues)
    issue_classes = sorted(L.issue_to_team)
    mask = L.team_issue_mask(issue_classes, team_classes)
    assert mask.shape == (len(team_classes), len(issue_classes))
    for i, iid in enumerate(issue_classes):
        t_row = team_classes.index(L.issue_to_team[iid])
        assert mask[t_row, i]


def test_all_11_teams_present():
    assert len(L.team_to_issues) == 11
    assert len(L.team_names) == 11
