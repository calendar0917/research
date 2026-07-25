#!/usr/bin/env bash
# 从 templates/track 初始化一个新课题目录。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$ROOT/templates/track"
TRACKS="$ROOT/tracks"

usage() {
  cat <<'EOF'
用法:
  ./scripts/init_track.sh <slug> [标题]

示例:
  ./scripts/init_track.sh ksvd "KSVD 结构字典学习"
  ./scripts/init_track.sh pe-ablation "位置编码消融"

slug: 小写字母/数字/连字符，如 gnn-gsn、ksvd
EOF
  exit 1
}

[[ $# -lt 1 ]] && usage
SLUG="$1"
TITLE="${2:-$SLUG}"

if [[ ! "$SLUG" =~ ^[a-z0-9]+([a-z0-9-]*[a-z0-9])?$ ]]; then
  echo "错误: slug 仅允许小写字母、数字、连字符: got '$SLUG'" >&2
  exit 1
fi

DEST="$TRACKS/$SLUG"
if [[ -e "$DEST" ]]; then
  echo "错误: 已存在 $DEST" >&2
  exit 1
fi

if [[ ! -d "$TEMPLATE" ]]; then
  echo "错误: 找不到模板 $TEMPLATE" >&2
  exit 1
fi

DATE="$(date +%Y-%m-%d)"
mkdir -p "$DEST"
cp -a "$TEMPLATE"/. "$DEST"/

# 替换占位符（TRACK.md 与 docs/README.md）
for f in "$DEST/TRACK.md" "$DEST/docs/README.md"; do
  if [[ -f "$f" ]]; then
    sed -i \
      -e "s/{{SLUG}}/${SLUG//\//\\/}/g" \
      -e "s/{{TITLE}}/${TITLE//\//\\/}/g" \
      -e "s/{{DATE}}/${DATE}/g" \
      "$f"
  fi
done

# 站点侧短入口
DOCS_TRACK="$ROOT/docs/tracks/${SLUG}.md"
mkdir -p "$ROOT/docs/tracks"
if [[ ! -f "$DOCS_TRACK" ]]; then
  cat >"$DOCS_TRACK" <<EOF
# ${TITLE}

- Track 手册：[TRACK.md](../../tracks/${SLUG}/TRACK.md)
- 状态：见 TRACK.md
- 全局进度：[research_guide.md](../research_guide.md)
EOF
fi

echo "已创建 track: $DEST"
echo "  编辑: $DEST/TRACK.md"
echo "  站点: $DOCS_TRACK"
echo "  下一步: 写清问题/范围/协议，再开始 code/"
