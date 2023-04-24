#!/bin/bash

function split_image() {
  file=$1
  filename=$(basename -- "$file")
  extension="${filename##*.}"
  filename="${filename%.*}"

  # 获取图片宽度和高度
  dimensions=$(identify -format "%wx%h" "$file")
  width=$(echo $dimensions | cut -d 'x' -f1)
  height=$(echo $dimensions | cut -d 'x' -f2)

  # 计算可以切割出的 512x512 图片的行数和列数
  num_rows=$((height / 512))
  num_cols=$((width / 512))

  # 如果可以至少切割出一张 512x512 的图片
  if [ $num_rows -gt 0 ] && [ $num_cols -gt 0 ]; then
    # 遍历每个切割点，切割图片
    for ((row=0; row<$num_rows; row++)); do
      for ((col=0; col<$num_cols; col++)); do
        convert "$file" -crop 512x512+$((col * 512))+$((row * 512)) "output/${filename}_${row}_${col}.${extension}"
      done
    done
  fi
}

mkdir output
export -f split_image

# 设置并发数
CONCURRENT=16

# 使用 parallel 处理 images 文件夹中的所有图片，并显示进度和预计剩余时间
find images -type f \( -iname "*.jpeg" -o -iname "*.png" \) | parallel --eta --progress -j $CONCURRENT split_image {}
