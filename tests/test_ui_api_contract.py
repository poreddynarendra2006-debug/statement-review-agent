"""The screens and the API have to agree on field names.

A reviewer pressing Acknowledge got "Failed to record action: Field required".
The screens were posting `action_type` and `notes`; the API has always taken
`status` and `note`, and accepts only three values. Nothing was ever recorded,
and the failure showed up only as a browser alert - no test, on either side,
compared the two.

Both sides pass their own tests happily, so these check the seam between them:
the names and the values the screens send are read straight out of the API's
own model, so changing one without the other fails here rather than in a demo.
"""

import json
import re
from pathlib import Path
from typing import get_args

import pytest

from api.models import ActionRequest

UI = Path(__file__).resolve().parents[1] / "ui" / "js"
FINDINGS = UI / "findings.js"


@pytest.fixture(scope="module")
def findings_js():
    if not FINDINGS.exists():
        pytest.skip("the screens are not in this checkout")
    return FINDINGS.read_text(encoding="utf-8")


def api_field_names():
    return set(ActionRequest.model_fields)


def api_statuses():
    return set(get_args(ActionRequest.model_fields["status"].annotation))


# --- the names ---------------------------------------------------------------


def test_the_screens_post_the_fields_the_api_requires(findings_js):
    posted = re.search(r"recordAction\([^)]*,\s*\{(.*?)\}", findings_js, re.S)
    assert posted, "could not find the call that records a reviewer's decision"

    sent = set(re.findall(r"(\w+)\s*:", posted.group(1)))
    required = {name for name, f in ActionRequest.model_fields.items() if f.is_required()}

    missing = required - sent
    assert not missing, f"the screens never send {sorted(missing)}, so the API rejects the call"

    unknown = sent - api_field_names()
    assert not unknown, f"the API has no field called {sorted(unknown)}"


def test_the_old_field_names_are_gone(findings_js):
    for stale in ("action_type", "notes:"):
        assert stale not in findings_js, f"{stale} is not part of the API's contract"


# --- the values --------------------------------------------------------------


def test_every_button_sends_a_value_the_api_accepts(findings_js):
    table = re.search(r"const ACTION_STATUS = \{(.*?)\}", findings_js, re.S)
    assert table, "the buttons no longer map to API values"

    sent = set(re.findall(r"'([A-Z_]+)'", table.group(1)))
    assert sent, "no values found in the button table"

    rejected = sent - api_statuses()
    assert not rejected, f"the API would refuse {sorted(rejected)}"


def test_all_three_decisions_are_offered(findings_js):
    table = re.search(r"const ACTION_STATUS = \{(.*?)\}", findings_js, re.S)
    sent = set(re.findall(r"'([A-Z_]+)'", table.group(1)))

    assert sent == api_statuses(), "a reviewer should be able to reach every decision the API records"


def test_a_recorded_decision_is_read_back_as_status(findings_js):
    # The list endpoint returns `status`; reading `action_type` off it showed
    # every reviewed finding as undefined.
    assert "action.status" in findings_js or "ACTION_LABEL[action.status]" in findings_js


# --- and that the API really does behave the way the screens now assume ------


def test_the_api_accepts_what_the_screens_now_send(findings_js):
    table = re.search(r"const ACTION_STATUS = \{(.*?)\}", findings_js, re.S)
    for value in re.findall(r"'([A-Z_]+)'", table.group(1)):
        request = ActionRequest(finding_ref="VAL_BS_01:Acme:2023", status=value,
                                note="Recorded on 2026-09-12T16:42:05.082Z",
                                reviewer="poojitha")
        assert request.status == value


def test_the_api_refuses_what_the_screens_used_to_send():
    with pytest.raises(Exception):
        ActionRequest(finding_ref="VAL_BS_01:Acme:2023", status="acknowledged")


def test_the_payload_the_screens_build_is_valid_json_for_the_api():
    # The exact shape from the browser, as it now stands.
    payload = {
        "finding_ref": "VAL_BS_01:Harborline Foods:2021",
        "status": "VERIFIED",
        "reviewer": "poojitha",
        "note": "Recorded on 2026-09-12T16:42:05.082Z",
    }

    request = ActionRequest(**json.loads(json.dumps(payload)))

    assert request.finding_ref and request.status == "VERIFIED"
