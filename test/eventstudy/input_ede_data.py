import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from datafeed import Datafeed
from datafeed.excel import apply_time_alignment
import pandas as pd
from pathlib import Path

df_market = Datafeed("daily_market")

# 选择操作模式
print("请选择数据导入模式：")
print("1. insert - 插入新数据（跳过重复，检测冲突）")
print("2. update - 更新已存在数据")
print("3. clear - 清空表中所有数据")
mode = input("请输入模式 (insert/update/clear): ").strip().lower()

if mode not in ['insert', 'update', 'clear']:
    print("✗ 无效的模式，请输入 insert、update 或 clear")
    df_market.close()
    exit(1)

# 如果是清空模式，执行清空并退出
if mode == 'clear':
    confirm = input("⚠ 警告：此操作将删除表中所有数据，确定要继续吗？(yes/no): ").strip().lower()
    if confirm != 'yes':
        print("✗ 操作已取消")
        df_market.close()
        exit(0)

    deleted_rows = df_market.truncate_table()
    print(f"✓ 表已清空，删除 {deleted_rows} 行数据")
    df_market.close()
    exit(0)

print(f"✓ 使用 {mode} 模式")

# 批量处理Wind长格式文件（CSV/Excel）
folder_path = Path(r'C:\Users\Janis\OneDrive\factor-frame\betalens\tests\eventstudy\data')
data_files = (
    list(folder_path.glob('*.csv')) +
    list(folder_path.glob('*.xls')) +
    list(folder_path.glob('*.xlsx'))
)

total_inserted = 0
total_skipped = 0
success_count = 0
error_count = 0

for data_file in data_files:
    try:
        # 1. 根据文件扩展名读取文件
        file_ext = data_file.suffix.lower()
        if file_ext == '.csv':
            df = pd.read_csv(data_file, encoding='gbk')
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(data_file)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")

        # 2. 重命名列（根据实际列名调整）
        df = df.rename(columns={
            '代码': 'code',
            '简称': 'name',
            '日期': 'date'
        })

        # 3. 转换为长格式
        # Wind导出已经是长格式，但数据在多列中，需要melt
        id_cols = ['code', 'name', 'date']
        value_cols = [col for col in df.columns if col not in id_cols]

        df_long = pd.melt(
            df,
            id_vars=id_cols,
            value_vars=value_cols,
            var_name='metric',
            value_name='value'
        )

        # 4. 应用时间对齐（根据metric类型设置不同时间）
        df_long = apply_time_alignment(
            df_long,
            date_column='date',
            metric_column='metric'
        )

        # 5. 重命名date为datetime
        df_long = df_long.rename(columns={'date': 'datetime'})

        # 6. 根据模式处理数据
        if mode == 'insert':
            # 插入模式：检测重复和冲突
            new_rows, skipped_rows, conflicts = df_market.insert_with_conflict_check(
                df_long,
                date_column='datetime',
                code_column='code',
                metric_column='metric'
            )

            # 打印冲突记录
            if conflicts:
                print(f"\n⚠ {data_file.name} 发现 {len(conflicts)} 条冲突记录（相同key但value不同）：")
                for conf in conflicts[:10]:  # 只显示前10条
                    print(f"  datetime={conf['datetime']}, code={conf['code']}, name={conf['name']}, "
                          f"metric={conf['metric']}, 数据库value={conf['db_value']}, 新value={conf['new_value']}")
                if len(conflicts) > 10:
                    print(f"  ... 还有 {len(conflicts)-10} 条冲突记录")

            total_inserted += new_rows
            total_skipped += skipped_rows
            success_count += 1
            print(f"✓ {data_file.name}: 插入{new_rows}行, 跳过{skipped_rows}行")

        else:  # update模式
            # 更新模式：使用SQL UPDATE
            updated_rows = df_market.update_data(
                df_long,
                date_column='datetime',
                code_column='code',
                metric_column='metric'
            )

            total_inserted += updated_rows
            success_count += 1
            print(f"✓ {data_file.name}: 更新{updated_rows}行")

    except Exception as e:
        error_count += 1
        print(f"✗ {data_file.name}: {str(e)}")
        import traceback
        traceback.print_exc()

print(f"\n处理完成: 成功{success_count}个, 失败{error_count}个")
if mode == 'insert':
    print(f"总计: 插入{total_inserted}行, 跳过{total_skipped}行")
else:
    print(f"总计: 更新/插入{total_inserted}行")

df_market.close()
