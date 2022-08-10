#!/usr/bin/env bash

set -Eeo pipefail

pre_run () {
  zenml integration install huggingface torch
}

pre_run_forced () {
  zenml integration install huggingface torch -y
}
