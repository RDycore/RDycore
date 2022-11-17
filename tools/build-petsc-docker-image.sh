#!/usr/bin/env bash

# This script builds a Docker image that contains an installation of PETSc
# configured specifically for RDycore. PETSc is installed in /opt/haero,
# on top of a recent Ubuntu image. Run it like so:
#
# ./build-petsc-docker-image.sh [petsc-hash]
#
# The arguments are:
# [petsc-hash] - A Git hash or tag identifying the revision of PETSc to build.u
#                If omitted, the HEAD of the repository is built.
#
# For this script to work, Docker must be installed on your machine.
PETSC_HASH=$1

if [[ "$1" == "" ]]; then
  PETSC_HASH=HEAD
fi

DOCKERHUB_USER=coherellc
IMAGE_NAME=rdycore-petsc
TAG=$PETSC_HASH

# Build the image locally.
mkdir -p docker-build
cp Dockerfile.petsc docker-build/Dockerfile
cd docker-build
docker build -t $IMAGE_NAME:$TAG --network=host \
  --build-arg PETSC_HASH=$PETSC_HASH \
  .
STATUS=$?
cd ..
rm -rd docker-build

if [[ "$STATUS" == "0" ]]; then
  # Tag the image.
  docker image tag $IMAGE_NAME:$TAG $DOCKERHUB_USER/$IMAGE_NAME:$TAG

  echo "To upload this image to DockerHub, use the following:"
  echo "docker login"
  echo "docker image push $DOCKERHUB_USER/$IMAGE_NAME:$TAG"
  echo "docker logout"
fi
