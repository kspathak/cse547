#!/bin/sh

docker run -p 8888:8888 -it --rm \
       -v "$HOME/.aws:/root/.aws" \
       -v "$(pwd)/data:/data" \
       -v "$(pwd):/local" \
       -m=4G \
       cse547:latest \
       "$@"
