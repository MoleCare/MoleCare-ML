"""Cross-file consistency checks for the deployment configuration.

canary_deploy.py and the deploy workflow have to agree on which Lambda function
they act on and in which region. They did not: the script defaulted to
"molecare-ml-prod" in us-east-1 while the workflows deploy "molecare-ml-production"
in eu-west-2. Nothing failed loudly -- the canary just looked for a function
that was not there.

These checks read the files as text rather than importing canary_deploy, so they
run without boto3 or wandb installed.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANARY = ROOT / "scripts" / "canary_deploy.py"
DEPLOY_WF = ROOT / ".github" / "workflows" / "ml-model-deploy.yml"
TRAIN_WF = ROOT / ".github" / "workflows" / "ml-model-train.yml"


def _default(source: str, name: str) -> str:
    """Extract the fallback in `NAME = os.environ.get("X", "fallback")`."""
    m = re.search(rf'^{name}\s*=\s*os\.environ\.get\([^,]+,\s*"([^"]+)"\)', source, re.M)
    assert m, f"{name} is not read from the environment with a default"
    return m.group(1)


def test_canary_targets_a_function_the_workflow_actually_deploys():
    default = _default(CANARY.read_text(), "FUNCTION_NAME")
    deployed = set(re.findall(r"molecare-ml-[a-z]+", DEPLOY_WF.read_text()))
    assert default in deployed, (
        f"canary_deploy.py defaults to {default!r}, but the workflow only ever "
        f"deploys {sorted(deployed)}"
    )


def test_canary_region_matches_the_workflow_region():
    default = _default(CANARY.read_text(), "AWS_REGION")
    m = re.search(r"^\s*AWS_REGION:\s*(\S+)", DEPLOY_WF.read_text(), re.M)
    assert m, "AWS_REGION not found in the deploy workflow"
    assert default == m.group(1), (
        f"canary_deploy.py defaults to region {default!r} but the workflow uses "
        f"{m.group(1)!r}; the canary would query the wrong region"
    )


def test_canary_target_is_overridable_from_the_command_line():
    src = CANARY.read_text()
    assert "--function-name" in src and "--region" in src
    assert "CanaryDeployer(args.function_name" in src, (
        "main() must use the parsed arguments, not the module-level constant"
    )


def test_workflows_do_not_cd_into_a_directory_that_is_never_created():
    """actions/checkout with no `path:` checks out to the workspace root.

    `cd MoleCare-ML` therefore fails with 'No such file or directory'. It was
    masked because the job died earlier at Configure AWS credentials.
    """
    for wf in (TRAIN_WF, DEPLOY_WF):
        code = "\n".join(
            line for line in wf.read_text().splitlines() if not line.lstrip().startswith("#")
        )
        assert not re.search(r"^\s*cd\s+MoleCare-ML\s*$", code, re.M), (
            f"{wf.name} cds into a directory the checkout never creates"
        )


def test_uploaded_artifact_paths_are_rooted_at_the_workspace():
    code = "\n".join(
        line for line in TRAIN_WF.read_text().splitlines() if not line.lstrip().startswith("#")
    )
    assert not re.search(r"^\s*MoleCare-ML/", code, re.M), (
        "artifact paths still point inside a directory that does not exist"
    )
