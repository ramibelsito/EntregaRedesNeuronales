#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Uso:
  ./clean_duplicate_images.sh ROOT_SPLIT [--apply] [--train-name train] [--eval-name val]

Ejemplo para tu estructura:
  ./clean_duplicate_images.sh ./data/Split_smol
  ./clean_duplicate_images.sh ./data/Split_smol --apply

Qué hace:
  1. Busca imágenes dentro de ROOT_SPLIT/train y ROOT_SPLIT/val.
  2. Calcula SHA256 de cada archivo, o sea detecta duplicados exactos por contenido.
  3. Para cada grupo de imágenes repetidas, conserva una sola copia.
  4. Los grupos repetidos conservados se asignan alternando:
       grupo duplicado 1 -> train
       grupo duplicado 2 -> val
       grupo duplicado 3 -> train
       ...
  5. Las copias sobrantes NO se borran: se mueven a ROOT_SPLIT/_duplicates_removed/<timestamp>/...
  6. Genera un reporte TSV con lo que encontró e hizo.

Por seguridad, por defecto corre en modo dry-run. Para modificar archivos usá --apply.

Notas:
  - Detecta duplicados byte-a-byte. Si la misma imagen fue recomprimida o redimensionada,
    no la va a marcar como duplicada.
  - Mantiene la carpeta de clase: train/<clase>/archivo.ext o val/<clase>/archivo.ext.
  - Si una misma imagen aparece con clases distintas, la saltea y lo deja marcado como conflicto.
EOF
}

ROOT=""
APPLY=0
TRAIN_NAME="train"
EVAL_NAME="val"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --train-name)
      TRAIN_NAME="${2:?Falta valor para --train-name}"
      shift 2
      ;;
    --eval-name|--val-name)
      EVAL_NAME="${2:?Falta valor para --eval-name}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$ROOT" ]]; then
        ROOT="$1"
        shift
      else
        echo "Argumento no reconocido: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$ROOT" ]]; then
  usage >&2
  exit 1
fi

ROOT="$(realpath "$ROOT")"
TRAIN_DIR="$ROOT/$TRAIN_NAME"
EVAL_DIR="$ROOT/$EVAL_NAME"

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "No existe la carpeta de train: $TRAIN_DIR" >&2
  exit 1
fi
if [[ ! -d "$EVAL_DIR" ]]; then
  echo "No existe la carpeta de evaluación: $EVAL_DIR" >&2
  echo "Si tu carpeta se llama distinto, usá por ejemplo: --eval-name evaluacion" >&2
  exit 1
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Falta el comando requerido: $1" >&2
    exit 1
  }
}

require_cmd find
require_cmd sha256sum
require_cmd sort
require_cmd awk
require_cmd realpath
require_cmd date
require_cmd mktemp

TS="$(date +%Y%m%d-%H%M%S)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

REPORT_DIR="$ROOT/_dedup_reports"
BACKUP_DIR="$ROOT/_duplicates_removed/$TS"
REPORT="$REPORT_DIR/duplicates_report_$TS.tsv"
HASHES="$TMP_DIR/hashes.tsv"
DUP_HASHES="$TMP_DIR/dup_hashes.txt"

# Extensiones típicas de imágenes. Agregá más si tu dataset usa otro formato.
find "$TRAIN_DIR" "$EVAL_DIR" -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) \
  -print0 > "$TMP_DIR/files0"

TOTAL_FILES=0
while IFS= read -r -d '' file; do
  hash="$(sha256sum -- "$file" | awk '{print $1}')"
  printf '%s\t%s\n' "$hash" "$file" >> "$HASHES"
  TOTAL_FILES=$((TOTAL_FILES + 1))
done < "$TMP_DIR/files0"

touch "$HASHES"
cut -f1 "$HASHES" | sort | uniq -d > "$DUP_HASHES"
DUP_GROUPS="$(wc -l < "$DUP_HASHES" | tr -d ' ')"
DUP_FILES="$(awk -F '\t' 'NR==FNR {dup[$1]=1; next} ($1 in dup) {c++} END {print c+0}' "$DUP_HASHES" "$HASHES")"
EXTRA_COPIES=$((DUP_FILES - DUP_GROUPS))

if [[ "$APPLY" -eq 1 ]]; then
  mkdir -p "$REPORT_DIR" "$BACKUP_DIR"
else
  mkdir -p "$REPORT_DIR"
fi

{
  printf 'hash\tstatus\ttarget_split\tclass\tkept_or_would_keep\tmoved_or_would_move\tnote\n'
} > "$REPORT"

split_of() {
  local p="$1"
  if [[ "$p" == "$TRAIN_DIR/"* ]]; then
    printf '%s\n' "$TRAIN_NAME"
  elif [[ "$p" == "$EVAL_DIR/"* ]]; then
    printf '%s\n' "$EVAL_NAME"
  else
    printf 'UNKNOWN\n'
  fi
}

class_of() {
  local p="$1"
  local rel
  if [[ "$p" == "$TRAIN_DIR/"* ]]; then
    rel="${p#"$TRAIN_DIR/"}"
  elif [[ "$p" == "$EVAL_DIR/"* ]]; then
    rel="${p#"$EVAL_DIR/"}"
  else
    printf 'UNKNOWN\n'
    return
  fi

  if [[ "$rel" == */* ]]; then
    printf '%s\n' "${rel%%/*}"
  else
    printf 'NO_CLASS_DIR\n'
  fi
}

relative_to_root() {
  local p="$1"
  printf '%s\n' "${p#"$ROOT/"}"
}

unique_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    printf '%s\n' "$path"
    return 0
  fi

  local dir base name ext candidate i
  dir="$(dirname -- "$path")"
  base="$(basename -- "$path")"

  if [[ "$base" == *.* ]]; then
    ext=".${base##*.}"
    name="${base%.*}"
  else
    ext=""
    name="$base"
  fi

  i=1
  while true; do
    candidate="$dir/${name}__dedup_${i}${ext}"
    if [[ ! -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    i=$((i + 1))
  done
}

same_file_path() {
  [[ "$(realpath -m "$1")" == "$(realpath -m "$2")" ]]
}

choose_representative() {
  local target_split="$1"
  shift
  local p

  # Preferir una copia que ya esté en el split destino para mover menos archivos.
  for p in "$@"; do
    if [[ "$(split_of "$p")" == "$target_split" ]]; then
      printf '%s\n' "$p"
      return 0
    fi
  done

  # Si no hay ninguna en el destino, usar la primera en orden lexicográfico.
  printf '%s\n' "$1"
}

if [[ "$DUP_GROUPS" -eq 0 ]]; then
  echo "No encontré duplicados exactos. Imágenes analizadas: $TOTAL_FILES"
  echo "Reporte: $REPORT"
  exit 0
fi

echo "Imágenes analizadas: $TOTAL_FILES"
echo "Grupos de duplicados exactos: $DUP_GROUPS"
echo "Archivos dentro de grupos duplicados: $DUP_FILES"
echo "Copias sobrantes que se moverían/quitarían del dataset: $EXTRA_COPIES"
if [[ "$APPLY" -eq 1 ]]; then
  echo "Modo: APPLY. Voy a modificar archivos. Backup de sobrantes: $BACKUP_DIR"
else
  echo "Modo: DRY-RUN. No voy a modificar archivos. Para aplicar: agregá --apply"
fi

echo "Reporte: $REPORT"
echo

group_index=0
while IFS= read -r hash; do
  mapfile -t paths < <(awk -F '\t' -v h="$hash" '$1 == h {print $2}' "$HASHES" | sort)

  # Validar que todas las copias tengan la misma clase.
  declare -A classes_seen=()
  for p in "${paths[@]}"; do
    classes_seen["$(class_of "$p")"]=1
  done

  if [[ "${#classes_seen[@]}" -ne 1 ]]; then
    note="CONFLICTO: mismo contenido con clases distintas; no se mueve automáticamente"
    for p in "${paths[@]}"; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$hash" "skipped_conflicting_classes" "-" "$(class_of "$p")" "-" "$(relative_to_root "$p")" "$note" >> "$REPORT"
    done
    echo "[SKIP] $hash -> aparece en clases distintas. Revisar reporte."
    unset classes_seen
    continue
  fi
  unset classes_seen

  class="$(class_of "${paths[0]}")"
  if [[ "$class" == "NO_CLASS_DIR" || "$class" == "UNKNOWN" ]]; then
    note="No pude inferir carpeta de clase; no se mueve automáticamente"
    for p in "${paths[@]}"; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$hash" "skipped_no_class" "-" "$class" "-" "$(relative_to_root "$p")" "$note" >> "$REPORT"
    done
    echo "[SKIP] $hash -> no pude inferir clase. Revisar reporte."
    continue
  fi

  if (( group_index % 2 == 0 )); then
    target_split="$TRAIN_NAME"
    target_base="$TRAIN_DIR"
  else
    target_split="$EVAL_NAME"
    target_base="$EVAL_DIR"
  fi
  group_index=$((group_index + 1))

  rep="$(choose_representative "$target_split" "${paths[@]}")"
  dest_dir="$target_base/$class"
  dest="$dest_dir/$(basename -- "$rep")"

  # Si el representante ya está en el destino correcto, queda ahí.
  # Si no, se mueve conservando el nombre salvo conflicto de nombre.
  final_rep="$rep"
  if [[ "$(split_of "$rep")" != "$target_split" ]]; then
    mkdir_target_msg=""
    if [[ -e "$dest" && ! "$(realpath -m "$dest")" == "$(realpath -m "$rep")" ]]; then
      dest="$(unique_path "$dest")"
    fi

    final_rep="$dest"
    if [[ "$APPLY" -eq 1 ]]; then
      mkdir -p "$dest_dir"
      mv -- "$rep" "$dest"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$hash" "keep" "$target_split" "$class" "$(relative_to_root "$final_rep")" "-" "representante conservado" >> "$REPORT"

  for p in "${paths[@]}"; do
    # No mover el representante original ni su nueva ubicación.
    if same_file_path "$p" "$rep" || same_file_path "$p" "$final_rep"; then
      continue
    fi

    backup_dest="$BACKUP_DIR/$(relative_to_root "$p")"
    if [[ "$APPLY" -eq 1 ]]; then
      if [[ -e "$p" ]]; then
        mkdir -p "$(dirname -- "$backup_dest")"
        if [[ -e "$backup_dest" ]]; then
          backup_dest="$(unique_path "$backup_dest")"
        fi
        mv -- "$p" "$backup_dest"
      fi
      status="moved_to_backup"
      moved_path="$(relative_to_root "$backup_dest")"
    else
      status="would_move_to_backup"
      moved_path="$(relative_to_root "$p")"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$hash" "$status" "$target_split" "$class" "$(relative_to_root "$final_rep")" "$moved_path" "copia sobrante" >> "$REPORT"
  done

  echo "[$target_split] $class -> conservar: $(relative_to_root "$final_rep") | copias del grupo: ${#paths[@]}"
done < "$DUP_HASHES"

echo
if [[ "$APPLY" -eq 1 ]]; then
  echo "Listo. Las copias sobrantes quedaron en: $BACKUP_DIR"
  echo "Reporte final: $REPORT"
  echo "Para verificar que no queden duplicados exactos, volvé a correr el script sin --apply."
else
  echo "Dry-run terminado. Revisá el reporte y, si está bien, corré:"
  echo "  ./$(basename "$0") '$ROOT' --apply --train-name '$TRAIN_NAME' --eval-name '$EVAL_NAME'"
fi
