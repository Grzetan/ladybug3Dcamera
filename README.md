# Rotate all images in dir to right

mkdir -p ./calibration/cam1rotated && for img in ./calibration/cam1/*; do ffmpeg -i "$img" -vf "transpose=1" "./calibration/cam1rotated/$(basename "$img")"; done

