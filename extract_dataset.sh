#!/bin/bash
# AnyText2 数据集解压脚本
# 使用方法: bash extract_dataset.sh [选项]
# bash extract_dataset.sh 

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DATASET_DIR="./dataset"
EXTRACT_JSON=true
EXTRACT_LAION=false
EXTRACT_OCR=false
EXTRACT_WUKONG=false
PARALLEL=false
DRY_RUN=false

# 打印帮助信息
print_help() {
    echo -e "${BLUE}AnyText2 数据集解压脚本${NC}"
    echo ""
    echo "使用方法: bash extract_dataset.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --json-only      只解压 JSON 配置文件"
    echo "  --laion          解压 LAION 数据 (约118GB)"
    echo "  --ocr            解压 OCR 数据 (约7.3GB)"
    echo "  --wukong         解压 Wukong 数据 (约75GB)"
    echo "  --all            解压所有数据"
    echo "  --parallel       并行解压 (加速)"
    echo "  --dry-run        模拟运行，不实际解压"
    echo "  -h, --help       显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  bash extract_dataset.sh --json-only        # 只解压 JSON"
    echo "  bash extract_dataset.sh --all --parallel   # 解压所有并并行"
    echo "  bash extract_dataset.sh --laion --ocr      # 解压 LAION 和 OCR"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --json-only)
            EXTRACT_JSON=true
            EXTRACT_LAION=false
            EXTRACT_OCR=false
            EXTRACT_WUKONG=false
            shift
            ;;
        --laion)
            EXTRACT_LAION=true
            shift
            ;;
        --ocr)
            EXTRACT_OCR=true
            shift
            ;;
        --wukong)
            EXTRACT_WUKONG=true
            shift
            ;;
        --all)
            EXTRACT_JSON=true
            EXTRACT_LAION=true
            EXTRACT_OCR=true
            EXTRACT_WUKONG=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# 检查数据集目录是否存在
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${RED}错误: 数据集目录不存在: $DATASET_DIR${NC}"
    exit 1
fi

cd "$DATASET_DIR"

# 检查磁盘空间
check_disk_space() {
    local required=$1
    local available=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')

    if [ "$available" -lt "$required" ]; then
        echo -e "${RED}错误: 磁盘空间不足！${NC}"
        echo "需要: ${required}GB, 可用: ${available}GB"
        exit 1
    fi
}

# 解压函数 - 解压到当前目录，自动创建同名子目录
extract_zip() {
    local zip_file=$1

    echo -e "${BLUE}解压: $zip_file${NC}"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[模拟] 将解压: $zip_file${NC}"
        return
    fi

    if [ ! -f "$zip_file" ]; then
        echo -e "${YELLOW}警告: 文件不存在: $zip_file${NC}"
        return
    fi

    # 直接解压，unzip 会自动创建与 zip 同名的目录
    # -o: 强制覆盖已存在的文件
    unzip -q -o "$zip_file"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 完成: $zip_file${NC}"
    else
        echo -e "${RED}✗ 失败: $zip_file${NC}"
    fi
}

# 打印配置
echo -e "${BLUE}===== 解压配置 =====${NC}"
echo "数据集目录: $DATASET_DIR"
echo "解压 JSON: $EXTRACT_JSON"
echo "解压 LAION: $EXTRACT_LAION"
echo "解压 OCR: $EXTRACT_OCR"
echo "解压 WUKONG: $EXTRACT_WUKONG"
echo "并行解压: $PARALLEL"
echo "模拟运行: $DRY_RUN"
echo -e "${BLUE}==================${NC}"
echo ""

# 1. 解压 JSON 配置文件
if [ "$EXTRACT_JSON" = true ]; then
    echo -e "${BLUE}===== 解压 JSON 配置文件 =====${NC}"

    if [ -f "anytext2_json_files.zip" ]; then
        check_disk_space 2
        extract_zip "anytext2_json_files.zip"
    else
        echo -e "${YELLOW}警告: anytext2_json_files.zip 不存在${NC}"
    fi
    echo ""
fi

# 2. 解压 LAION 数据
if [ "$EXTRACT_LAION" = true ]; then
    echo -e "${BLUE}===== 解压 LAION 数据 =====${NC}"

    if [ -d "laion" ]; then
        cd laion

        if [ "$PARALLEL" = true ]; then
            echo -e "${YELLOW}使用并行解压...${NC}"
            check_disk_space 120

            for i in {1..5}; do
                if [ -f "laion_p${i}.zip" ]; then
                    if [ "$DRY_RUN" = true ]; then
                        echo -e "${YELLOW}[模拟] 将解压: laion_p${i}.zip${NC}"
                    else
                        # 创建目标目录并解压到其中
                        mkdir -p "laion_p${i}"
                        unzip -q -o "laion_p${i}.zip" -d "laion_p${i}" > /dev/null &
                    fi
                fi
            done
            wait  # 等待所有后台任务完成
            echo -e "${GREEN}✓ LAION 解压完成${NC}"
        else
            check_disk_space 120
            for i in {1..5}; do
                if [ -f "laion_p${i}.zip" ]; then
                    echo -e "${BLUE}解压: laion_p${i}.zip${NC}"
                    if [ "$DRY_RUN" = false ]; then
                        mkdir -p "laion_p${i}"
                        unzip -q -o "laion_p${i}.zip" -d "laion_p${i}"
                        echo -e "${GREEN}✓ 完成: laion_p${i}.zip${NC}"
                    fi
                fi
            done
        fi

        cd ..
    else
        echo -e "${YELLOW}警告: laion 目录不存在${NC}"
    fi
    echo ""
fi

# 3. 解压 OCR 数据
if [ "$EXTRACT_OCR" = true ]; then
    echo -e "${BLUE}===== 解压 OCR 数据 =====${NC}"

    if [ -d "ocr_data" ]; then
        cd ocr_data

        check_disk_space 8

        # OCR 数据集列表
        ocr_datasets=("Art" "COCO_Text" "LSVT" "MTWI2018" "ReCTS" "icdar2017rctw" "mlt2019")

        if [ "$PARALLEL" = true ]; then
            echo -e "${YELLOW}使用并行解压...${NC}"

            for dataset in "${ocr_datasets[@]}"; do
                if [ -d "$dataset" ]; then
                    (
                        cd "$dataset"
                        for zip_file in *.zip; do
                            if [ -f "$zip_file" ]; then
                                if [ "$DRY_RUN" = true ]; then
                                    echo -e "${YELLOW}[模拟] 将解压: $dataset/$zip_file${NC}"
                                else
                                    # 创建与 zip 同名的目录并解压到其中
                                    dir_name="${zip_file%.zip}"
                                    mkdir -p "$dir_name"
                                    unzip -q -o "$zip_file" -d "$dir_name" > /dev/null
                                fi
                            fi
                        done
                        echo -e "${GREEN}✓ 完成: $dataset${NC}"
                    ) &
                fi
            done
            wait
        else
            for dataset in "${ocr_datasets[@]}"; do
                if [ -d "$dataset" ]; then
                    echo -e "${BLUE}处理: $dataset${NC}"
                    cd "$dataset"
                    for zip_file in *.zip; do
                        if [ -f "$zip_file" ]; then
                            if [ "$DRY_RUN" = false ]; then
                                # 创建与 zip 同名的目录并解压到其中
                                dir_name="${zip_file%.zip}"
                                echo -e "${BLUE}解压: $zip_file${NC}"
                                mkdir -p "$dir_name"
                                unzip -q -o "$zip_file" -d "$dir_name"
                                echo -e "${GREEN}✓ 完成: $zip_file${NC}"
                            fi
                        fi
                    done
                    cd ..
                fi
            done
        fi

        cd ..
    else
        echo -e "${YELLOW}警告: ocr_data 目录不存在${NC}"
    fi
    echo ""
fi

# 4. 解压 Wukong 数据
if [ "$EXTRACT_WUKONG" = true ]; then
    echo -e "${BLUE}===== 解压 Wukong 数据 =====${NC}"

    check_disk_space 80

    # Wukong 数据集列表（5个分片）
    wukong_datasets=("wukong_1of5" "wukong_2of5" "wukong_3of5" "wukong_4of5" "wukong_5of5")

    if [ "$PARALLEL" = true ]; then
        echo -e "${YELLOW}使用并行解压...${NC}"

        for wukong in "${wukong_datasets[@]}"; do
            if [ -d "$wukong" ]; then
                (
                    cd "$wukong"
                    for zip_file in *.zip; do
                        if [ -f "$zip_file" ]; then
                            if [ "$DRY_RUN" = true ]; then
                                echo -e "${YELLOW}[模拟] 将解压: $wukong/$zip_file${NC}"
                            else
                                # 创建与 zip 同名的目录并解压到其中
                                dir_name="${zip_file%.zip}"
                                mkdir -p "$dir_name"
                                unzip -q -o "$zip_file" -d "$dir_name" > /dev/null
                            fi
                        fi
                    done
                    echo -e "${GREEN}✓ 完成: $wukong${NC}"
                ) &
            fi
        done
        wait
    else
        for wukong in "${wukong_datasets[@]}"; do
            if [ -d "$wukong" ]; then
                echo -e "${BLUE}处理: $wukong${NC}"
                cd "$wukong"
                for zip_file in *.zip; do
                    if [ -f "$zip_file" ]; then
                        if [ "$DRY_RUN" = false ]; then
                            # 创建与 zip 同名的目录并解压到其中
                            dir_name="${zip_file%.zip}"
                            echo -e "${BLUE}解压: $zip_file${NC}"
                            mkdir -p "$dir_name"
                            unzip -q -o "$zip_file" -d "$dir_name"
                            echo -e "${GREEN}✓ 完成: $zip_file${NC}"
                        fi
                    fi
                done
                cd ..
            fi
        done
    fi
    echo ""
fi

# 完成
echo -e "${GREEN}===== 解压完成 =====${NC}"
echo "可以使用以下命令查看解压结果:"
echo "  du -sh $DATASET_DIR/*/  # 查看各目录大小"
echo "  ls -lh $DATASET_DIR/    # 查看文件列表"
