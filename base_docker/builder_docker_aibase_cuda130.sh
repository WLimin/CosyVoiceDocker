#!/bin/bash
docker build --progress=plain \
 -t aibase:torch291_cuda130 \
-f Dockerfile.aibase_cuda130 \
./

