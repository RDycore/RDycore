#!/usr/bin/env bash

# This script builds a Docker image that contains an installation of PETSc
# configured specifically for RDycore. PETSc is installed in /usr/local/petsc,
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

# Build the image locally in configurations that use both 32-bit and 64-bit
# integers for PETSc IDs.
mkdir -p docker-build
cp Dockerfile.petsc docker-build/Dockerfile
cd docker-build
for id_type in int32 int64
do
  TAG=$PETSC_HASH-$id_type
  docker buildx build -t $IMAGE_NAME:$TAG --network=host \
    --build-arg PETSC_HASH=$PETSC_HASH \
    --build-arg PETSC_ID=$id_type \
    .
  STATUS=$?
  if [[ "$STATUS" == "0" ]]; then
    # Tag the image.
    docker image tag $IMAGE_NAME:$TAG $DOCKERHUB_USER/$IMAGE_NAME:$TAG
  fi
done
cd ..
rm -rd docker-build

if [[ "$STATUS" == "0" ]]; then
  # Tag the image.
  docker image tag $IMAGE_NAME:$TAG $DOCKERHUB_USER/$IMAGE_NAME:$TAG

  echo "To upload the images to DockerHub, use the following:"
  echo "docker login"
  for id_type in int32 int64
  do
    TAG=$PETSC_HASH-$id_type
    echo "docker image push $DOCKERHUB_USER/$IMAGE_NAME:$TAG"
  done
  echo "docker logout"
fi
