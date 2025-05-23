#!/bin/bash
set -e

# Automatically clean up loop device on exit
cleanup() {
  if [[ -n "$LOOP" ]] && [ -e "$LOOP" ] && losetup | grep -q "$LOOP"; then
    echo "⚠️ Cleaning up loop device $LOOP..."
    sudo losetup -d "$LOOP" || true
  fi
}

# Check required dependencies
check_dependencies() {
  local missing=()
  local deps=("dd" "losetup" "parted" "gdisk" "fdisk" "e2fsck" "resize2fs" \
              "truncate" "stat" "zip" "numfmt")

  for cmd in "${deps[@]}"; do
    if ! command -v "$cmd" &>/dev/null; then
      missing+=("$cmd")
    fi
  done

  if [ ${#missing[@]} -ne 0 ]; then
    echo "❌ Missing required tools:"
    for cmd in "${missing[@]}"; do
      echo "  - $cmd"
    done
    echo "👉 Please install with:"
    echo "   sudo apt-get install -y ${missing[*]}"
    exit 1
  fi
}

# Prompt if the .zip already exists
check_file_existed() {
  local zip="${1}.zip"
  if [ -f "$zip" ]; then
    echo "⚠️ File already exists: $zip"
    read -p "Delete and continue? (y/N) " confirm
    case "$confirm" in
      [Yy]*) echo "🗑️ Deleting $zip..."; rm -f "$zip" ;;
      *)     echo "↩️ Cancelled by user."; exit 0 ;;
    esac
  fi
}

# Truncate image file to match partition end + 1MB buffer
truncate_image_to_partition_end() {
  local img="$1"
  echo ">> Truncating image to match last partition end..."
  local end_byte
  end_byte=$(sudo parted "$img" unit B print | awk '/^ 1/ { gsub("B","",$3); print $3 }')
  local buffer=1048576
  local new_size=$(( end_byte + buffer ))

  echo "   Truncating to $new_size (with buffer)"
  sudo truncate -s "$new_size" "$img"

  echo ">> Aligning image to 512-byte boundary"
  local aligned_size=$(( ( $(stat -c %s "$img") / 512 + 1 ) * 512 ))
  sudo truncate -s "$aligned_size" "$img"
}

# Repair GPT backup header
fix_gpt_header() {
  local img="$1"
  echo ">> Fixing GPT backup header..."
  sudo gdisk "$img" <<EOF
w
y
EOF

  if [ $? -eq 0 ]; then
    echo "   ✅ GPT fix complete"
  else
    echo "   ❌ GPT fix failed"
    exit 1
  fi
  sudo gdisk -l "$img"
  sudo fdisk -l "$img"
}

# Compress and delete .img file
compress_and_cleanup() {
  local img="$1"
  local zip="${img}.zip"

  local img_size
  img_size=$(stat -c %s "$img")
  echo "📦 Image size: $img_size bytes ($(numfmt --to=iec --suffix=B "$img_size"))"

  echo ">> Compressing to $zip..."
  rm -f "$zip"
  if zip -9 -v "$zip" "$img"; then
    local zip_size
    zip_size=$(stat -c %s "$zip")
    echo "✅ Compression complete: $zip"
    echo "🗜️ Compressed size: $zip_size bytes ($(numfmt --to=iec --suffix=B "$zip_size"))"
    echo "🗑️ Deleting $img..."
    rm -f "$img"
    echo "✅ Done"
  else
    echo "❌ Compression failed"
    exit 1
  fi
}

#
# Main Script Execution
#

trap cleanup EXIT

if [ $# -ne 2 ]; then
  echo "❌ Missing arguments!"
  echo "Usage: $0 <base filename> <device like /dev/sdX>"
  echo
  echo "Available devices (likely USB or SD card):"
  lsblk -dpno NAME,SIZE,MODEL | grep -Ev 'loop|nvme|sda|mmcblk|boot' || true
  echo "👉 Insert SD card and try again."
  exit 1
fi

RAW_BASENAME="$1"
DEVICE="$2"
BASENAME="${RAW_BASENAME%%.*}"
IMG="${BASENAME}.img"

check_dependencies
check_file_existed "$IMG"

echo ">> Backing up $DEVICE to $IMG..."
sudo dd if="$DEVICE" of="$IMG" bs=64M status=progress conv=fsync

IMG_SIZE=$(stat -c %s "$IMG")
THRESHOLD_DEV=$((59 * 1024 * 1024 * 1024))

echo ">> Attaching loop device..."
LOOP=$(sudo losetup -Pf --show "$IMG")
echo "   Mounted as $LOOP"

echo ">> Partition layout:"
printf 'F\n' | sudo parted "$LOOP" ---pretend-input-tty unit B print

PART1_END_BYTE=$(sudo parted "$LOOP" unit B print | awk '/^ 1/ { gsub("B","",$3); print $3 }')
THRESHOLD_IMG=$((57 * 1024 * 1024 * 1024))

if [ "$PART1_END_BYTE" -le "$THRESHOLD_IMG" ]; then
  echo "   ✅ Partition < 57GiB. Skip resize, truncate + GPT repair only."
  sudo losetup -d "$LOOP"
  truncate_image_to_partition_end "$IMG"
  fix_gpt_header "$IMG"
  compress_and_cleanup "$IMG"
  exit 0
fi

SHRINK_GB=8
echo ">> Shrinking main partition by ${SHRINK_GB}GB..."

SECTOR_SIZE=512
BLOCK_SIZE=4096
SHRINK_SIZE=$((1024 * 1024 * 1024 * SHRINK_GB))

START_BYTE=$(sudo parted "$LOOP" unit B print | \
             awk '/^ 1/ { gsub("B","",$2); print $2 }')
END_BYTE=$(sudo parted "$LOOP" unit B print | \
           awk '/^ 1/ { gsub("B","",$3); print $3 }')

AVAILABLE_SPACE=$(( END_BYTE - START_BYTE ))
if [ "$AVAILABLE_SPACE" -lt "$SHRINK_SIZE" ]; then
  echo "⚠️  Partition is smaller than $SHRINK_GB GiB (only $(( AVAILABLE_SPACE / 1024 / 1024 / 1024 )) GiB available)"
  echo "↪️  Skipping shrink and GPT repair. Proceeding to compression..."
  sudo losetup -d "$LOOP"
  compress_and_cleanup "$IMG"
  exit 0
fi

TARGET_END_BYTE=$(( END_BYTE - SHRINK_SIZE ))
PART_LEN_BYTE=$(( TARGET_END_BYTE - START_BYTE + 1 ))
TARGET_BLOCKS=$(( PART_LEN_BYTE / BLOCK_SIZE ))

echo "   Start: $START_BYTE"
echo "   End (before): $END_BYTE"
echo "   End (after):  $TARGET_END_BYTE"
echo "   Length (bytes): $PART_LEN_BYTE"
echo "   Target blocks: $TARGET_BLOCKS"

# Use p1 by default, or try auto-detect if set
#USE_AUTO_FIND_EXT4=true
LOOP_EXT4_DEV="${LOOP}p1"

if [ "$USE_AUTO_FIND_EXT4" = true ]; then
  EXT4_PART=$(lsblk -ln -o NAME,FSTYPE "$LOOP" | awk '$2 == "ext4" { print $1; exit }')
  LOOP_EXT4_DEV="/dev/${EXT4_PART}"
  if [ ! -e "$LOOP_EXT4_DEV" ]; then
    echo "❌ ext4 partition not found"
    exit 1
  fi
fi

echo ">> Running fsck..."
sudo e2fsck -f -y -v -C 0 "$LOOP_EXT4_DEV"

echo ">> resize2fs to $TARGET_BLOCKS blocks..."
sudo resize2fs -p "$LOOP_EXT4_DEV" "$TARGET_BLOCKS"

TARGET_END_SECTOR=$(( (TARGET_END_BYTE + SECTOR_SIZE - 1) / SECTOR_SIZE ))
echo ">> Resizing partition with parted to sector $TARGET_END_SECTOR..."
sudo parted ---pretend-input-tty "$LOOP" <<EOF
resizepart 1 ${TARGET_END_SECTOR}s
Yes
quit
EOF

echo ">> Final fsck (just to be safe)..."
sudo e2fsck -f -y -v "$LOOP_EXT4_DEV"

echo ">> Checking final partition layout..."
sudo parted "$LOOP" unit s print | grep "^ 1"

echo ">> Detaching loop device..."
sudo losetup -d "$LOOP"

truncate_image_to_partition_end "$IMG"
fix_gpt_header "$IMG"
compress_and_cleanup "$IMG"
