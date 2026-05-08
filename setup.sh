#!/bin/sh
# Install the project's git hooks. Run once after cloning.
set -e
git config core.hooksPath hooks
echo "git hooks path set to hooks/"
