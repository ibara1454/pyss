#!/bin/sh

if [ ! -d $2 ]; then
  git clone $1 $2
fi
