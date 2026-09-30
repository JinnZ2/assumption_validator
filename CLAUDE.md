# assumption_validator

> A universal framework for detecting when the assumptions underlying any model, prediction, or decision system are breaking down.

Source: README.md

<!-- clone-refspec-note v1.1 -->
## Cloning and pushing
Shallow clones are single-branch by default.
Before pushing any branch other than the default
branch, run:

    git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
    git fetch --depth 1

Or clone with: git clone --depth 1 --no-single-branch <url>
Without this, the first push of a new branch
fails the tracking-ref check even when the
commit landed.
<!-- /clone-refspec-note v1.1 -->
