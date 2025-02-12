from urllib.parse import quote_plus

import pytest

from mlem.utils.github import (
    get_github_kwargs,
    github_check_rev,
    ls_branches,
    ls_github_branches,
)
from tests.conftest import (
    MLEM_TEST_REPO,
    MLEM_TEST_REPO_NAME,
    MLEM_TEST_REPO_ORG,
    long,
    need_test_repo_auth,
    need_test_repo_ssh_auth,
)


@long
@need_test_repo_ssh_auth
def test_ls_branches():
    assert "main" in ls_branches(MLEM_TEST_REPO)


@long
@need_test_repo_auth
def test_ls_github_branches():
    assert "main" in ls_github_branches(
        MLEM_TEST_REPO_ORG, MLEM_TEST_REPO_NAME
    )


@pytest.fixture
def set_mock_refs(mocker):
    def set(rev):
        mocker.patch(
            "mlem.utils.github._ls_github_refs",
            return_value={rev: ""},
        )

    return set


@pytest.mark.parametrize(
    "uri, rev",
    [
        ("https://github.com/org/repo/tree/simple_ref/path", "simple_ref"),
        (
            "https://github.com/org/repo/tree/ref/with/slashes/path",
            "ref/with/slashes",
        ),
    ],
)
def test_get_github_kwargs(set_mock_refs, uri, rev):
    set_mock_refs(rev)
    assert get_github_kwargs(uri) == {
        "org": "org",
        "repo": "repo",
        "path": "path",
        "sha": quote_plus(rev),
    }


def test_get_github_kwargs__empty_path(set_mock_refs):
    set_mock_refs("ref")
    assert get_github_kwargs("https://github.com/org/repo/tree/ref/") == {
        "org": "org",
        "repo": "repo",
        "path": "",
        "sha": "ref",
    }


@long
def test_github_check_rev():
    assert github_check_rev(
        MLEM_TEST_REPO_ORG, MLEM_TEST_REPO_NAME, "main"
    )  # branch
    assert not github_check_rev(
        MLEM_TEST_REPO_ORG, MLEM_TEST_REPO_NAME, "_____"
    )  # not exists
    assert github_check_rev(
        MLEM_TEST_REPO_ORG,
        MLEM_TEST_REPO_NAME,
        "bf022746331ec6888e58b483fbc1fb08313dffc0",
    )  # commit
    assert github_check_rev(
        MLEM_TEST_REPO_ORG, MLEM_TEST_REPO_NAME, "first_rev_link"
    )  # tag
