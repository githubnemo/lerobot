#!/bin/sh


EXT_CAMERA=$(v4l2-ctl --list-devices | grep 'Webcam C920' -A1 | tail -1 | tr -d "\t")

v4l2-ctl -d "${EXT_CAMERA}" -c focus_automatic_continuous=0
v4l2-ctl -d "${EXT_CAMERA}" -c focus_absolute=19
v4l2-ctl -d "${EXT_CAMERA}" -c saturation=92
