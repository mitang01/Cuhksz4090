# =============================================================================
# 多脑区单神经元 Spike Sorting — v4
# 更新: 2026-04-12
#
# 脑区通道映射 (1-indexed):
#   ATL:      Ch 65–80   → Python idx [64:80]
#   HG:       Ch 81–96   → Python idx [80:96]
#   VMPFC:    Ch 97–112  → Python idx [96:112]
#   Amygdala: Ch 113–128 → Python idx [112:128]
# =============================================================================


# ██████████████████████████████████████████████████████████████████████████
# BLOCK 0 ── 环境初始化与全局参数
# ██████████████████████████████████████████████████████████████████████████

import numpy as np
import shutil
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import spikeinterface.full as si
import spikeinterface.widgets as sw
from probeinterface import generate_linear_probe
import os
import warnings
warnings.simplefilter("ignore")

%matplotlib inline

# ── 并行参数 ──────────────────────────────────────────────────────────────
n_jobs     = max(1, os.cpu_count() - 2)
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
si.set_global_job_kwargs(**job_kwargs)
print(f"✅ 并行参数: n_jobs = {n_jobs}")

# ── 路径配置（story_listen 批处理）────────────────────────────────────────
DATASET_RUNS = [
    {
        "name": "sub5_story_listen",
        "recording_path": Path("/share/workspace3/ieeg/micro/story_listen_v1/sub5_story_listen_merged"),
        "results_base": Path("/share/home/mitan/Cuhksz4090/spike_sorting/mountainsort4/sub5_story_listen_sorting_results"),
    },
    {
        "name": "sub6_story_listen",
        "recording_path": Path("/share/workspace3/ieeg/micro/story_listen_v1/sub6_story_listen_merged"),
        "results_base": Path("/share/home/mitan/Cuhksz4090/spike_sorting/mountainsort4/sub6_story_listen_sorting_results"),
    },
]
# 兼容后续 Block 中对 base_folder 的引用（仅作为占位目录）。
base_folder = Path("/share/workspace3/ieeg/micro/story_listen_v1")

# ── 录制参数 ──────────────────────────────────────────────────────────────
SAMPLING_RATE      = 30000.0   # Hz
NUM_CHANNELS_TOTAL = 128       # 通道编号最大到128，总数为128
GAIN_TO_UV         = 0.195     # Intan 标准 ADC 转换系数

# ── 脑区通道映射（Python 0-indexed 左闭右开）─────────────────────────────
# 转换公式: 1-indexed [A, B]  →  0-indexed [A-1, B]
REGION_CHANNEL_MAP = {
    'ATL'      : (64,  80),    # Ch 65–80
    'HG'       : (80,  96),    # Ch 81–96
    'VMPFC'    : (96,  112),   # Ch 97–112
    'Amygdala' : (112, 128),   # Ch 113–128
}

# ── 质控阈值（可按需调整）────────────────────────────────────────────────
THRESH_SNR            = 5.0    # SNR 下限
THRESH_ISI_RATIO      = 0.5    # ISI 违规率上限（>0.5 视为 MUA）
THRESH_MIN_SPIKES     = 50     # 最少 spike 数
THRESH_MIN_FR         = 0.1    # 最低平均放电率 (Hz)
THRESH_PRESENCE_RATIO = 0.3    # 最低 presence ratio

# ── 可视化配色（每脑区一个颜色）─────────────────────────────────────────
REGION_COLORS = {
    'ATL'      : '#2196F3',    # 蓝
    'HG'       : '#4CAF50',    # 绿
    'VMPFC'    : '#FF9800',    # 橙
    'Amygdala' : '#E91E63',    # 红
}

print("✅ BLOCK 0 全局参数初始化完成")
print("   目标批处理数据集:")
for _job in DATASET_RUNS:
    print(f"     - {_job['name']}: {_job['recording_path']}")
    print(f"       输出目录: {_job['results_base']}")
print(f"   脑区列表     : {list(REGION_CHANNEL_MAP.keys())}")


# ██████████████████████████████████████████████████████████████████████████
# BLOCK 1 ── 加载完整二进制数据（只加载一次，后续按脑区切片）
# ██████████████████████████████████████████████████████████████████████████

# 批处理入口：保持 BLOCK 1-3 主体排序逻辑不变，仅按不同输入/输出路径重复执行。
if "__SPIKESORT_CXD_BATCH_EXECUTING__" not in globals():
    globals()["__SPIKESORT_CXD_BATCH_EXECUTING__"] = True

    src_text = Path(__file__).read_text(encoding="utf-8")
    block_match = re.search(
        r"print\(\"\\n\" \+ \"=\"\*65\)\n"
        r"print\(\"BLOCK 1: 加载完整二进制数据\"\)\n"
        r"print\(\"=\"\*65\)\n"
        r"[\s\S]*?print\(\"✅ BLOCK 3 完成！\"\)\n"
        r"print\(\"=\"\*65\)\n",
        src_text,
    )
    if block_match is None:
        raise RuntimeError("无法定位 BLOCK 1-3 代码段，批处理终止。")
    block_1_to_3 = block_match.group(0)

    for run_i, run_cfg in enumerate(DATASET_RUNS, start=1):
        recording_path = Path(run_cfg["recording_path"])
        results_base = Path(run_cfg["results_base"])
        results_base.mkdir(parents=True, exist_ok=True)

        print("\n" + "▓" * 72)
        print(f"🚀 批处理任务 {run_i}/{len(DATASET_RUNS)}: {run_cfg['name']}")
        print(f"   输入 merged 文件: {recording_path}")
        print(f"   输出目录: {results_base}")
        print("▓" * 72)

        exec(block_1_to_3, globals(), globals())

    print("\n✅ 所有 story_listen 批处理任务已完成（sub5 + sub6）。")
    raise SystemExit(0)

print("\n" + "="*65)
print("BLOCK 1: 加载完整二进制数据")
print("="*65)

recording_full = si.read_binary(
    file_paths=str(recording_path),
    num_channels=NUM_CHANNELS_TOTAL,
    dtype='int16',
    sampling_frequency=SAMPLING_RATE,
    gain_to_uV=GAIN_TO_UV,
    time_axis=0
)

total_duration_s = recording_full.get_num_samples() / SAMPLING_RATE
print(f"✅ 数据加载成功!")
print(f"   通道数   : {NUM_CHANNELS_TOTAL}")
print(f"   采样率   : {SAMPLING_RATE:.0f} Hz")
print(f"   总时长   : {total_duration_s:.1f} 秒  ({total_duration_s / 60:.1f} 分钟)")
print(f"   总样本数 : {recording_full.get_num_samples():,}")


# ██████████████████████████████████████████████████████████████████████████
# BLOCK 2 ── 主循环：逐脑区 Sorting（核心）
# ██████████████████████████████████████████████████████████████████████████

# ── 全局容器，汇总所有脑区结果 ───────────────────────────────────────────
all_spike_times = {}   # { 'ATL_unit1': np.array(秒), ... }
all_units_meta  = []   # 每个 unit 的元数据行

for region, (ch_start, ch_end) in REGION_CHANNEL_MAP.items():

    n_ch = ch_end - ch_start    # 每脑区 16 个通道

    print(f"\n{'█'*65}")
    print(f"  🧠  脑区 : {region}")
    print(f"      通道 (1-indexed) : Ch{ch_start + 1} – Ch{ch_end}")
    print(f"      通道 (0-indexed) : [{ch_start} : {ch_end}]")
    print(f"      通道数           : {n_ch}")
    print(f"{'█'*65}")

    # ── 2.1 通道切片 ──────────────────────────────────────────────────────
    all_ch_ids    = recording_full.get_channel_ids()
    target_ch_ids = all_ch_ids[ch_start:ch_end]
    recording_sub = recording_full.select_channels(channel_ids=target_ch_ids)
    print(f"\n  [2.1] ✅ 通道切片完成: {recording_sub.get_num_channels()} 个通道")

    # ── 2.2 虚拟探针 + 独立 group（单电极关键步骤）───────────────────────
    # ypitch=1000 μm：拉开物理距离，防止算法跨通道聚类
    # group=每通道独立：MountainSort4 将每个通道视为独立电极
    probe = generate_linear_probe(num_elec=n_ch, ypitch=1000)
    probe.set_device_channel_indices(np.arange(n_ch))
    recording_sub = recording_sub.set_probe(probe)
    recording_sub.set_channel_groups(np.arange(n_ch))
    print(f"  [2.2] ✅ 虚拟探针挂载完成，{n_ch} 个通道独立分组")

    # ── 2.3 预处理：带通滤波 + 共模参考 ──────────────────────────────────
    # bandpass: 去除 LFP 低频 (<300Hz) 和高频噪声 (>6000Hz)
    # CMR:      去除脑区内共享背景噪声（咀嚼/碰撞/电磁干扰等）
    recording_f   = si.bandpass_filter(recording_sub, freq_min=300, freq_max=6000)
    recording_cmr = si.common_reference(recording_f, reference='global', operator='median')
    print(f"  [2.3] ✅ 带通滤波 [300–6000 Hz] + CMR 全局中位数参考完成")

    # ── 2.4 MountainSort4 Spike Sorting ───────────────────────────────────
    sort_dir = results_base / f'{region}_sorting'
    if sort_dir.exists():
        shutil.rmtree(sort_dir)

    print(f"\n  [2.4] 🔄 运行 MountainSort4 ...")
    print(f"        输出目录: {sort_dir}")

    sorting = si.run_sorter(
        sorter_name='mountainsort4',
        recording=recording_cmr,
        folder=str(sort_dir),
        verbose=True,
        detect_threshold=5,    # 5 倍标准差
        freq_min=300,
        freq_max=6000,
        adjacency_radius=-1,   # 完全不跨通道聚类
    )

    n_units = len(sorting.get_unit_ids())
    print(f"  [2.4] ✅ Sorting 完成！{region} 共找到 {n_units} 个初始聚类")

    # ── 2.5 SortingAnalyzer：计算所有扩展组件 ────────────────────────────
    analyzer_dir = results_base / f'{region}_analyzer'
    if analyzer_dir.exists():
        shutil.rmtree(analyzer_dir)

    print(f"\n  [2.5] 🔄 创建 SortingAnalyzer ...")
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording_cmr,
        format="binary_folder",
        folder=str(analyzer_dir)
    )

    # 按依赖顺序依次计算（顺序不能打乱）
    print(f"        → random_spikes ...")
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)

    print(f"        → waveforms ...")
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)

    print(f"        → templates ...")
    analyzer.compute("templates")

    print(f"        → noise_levels ...")
    analyzer.compute("noise_levels")

    print(f"        → spike_amplitudes ...")
    analyzer.compute("spike_amplitudes")

    print(f"        → unit_locations ...")
    analyzer.compute("unit_locations")

    print(f"        → correlograms ...")
    analyzer.compute("correlograms")

    print(f"        → template_similarity ...")
    analyzer.compute("template_similarity")

    # ── quality_metrics：动态列名适配，永久兼容新旧版本 ──────────────────
    print(f"        → quality_metrics (动态列名适配) ...")

    from spikeinterface.metrics.quality.quality_metrics import ComputeQualityMetrics
    available_metrics = ComputeQualityMetrics.get_available_metric_names()
    print(f"          当前环境可用 metrics: {available_metrics}")

    # 候选列名：每项格式 (逻辑key, [优先级从高到低的候选名])
    METRIC_CANDIDATES = [
        ('num_spikes',     ['num_spikes']),
        ('snr',            ['snr']),
        ('isi',            ['isi_violations_ratio',    # 新版输入名
                            'isi_violation']),          # 旧版输入名
        ('isi_count',      ['isi_violations_count']),
        ('rp',             ['rp_contamination',        # 新版
                            'rp_violation']),           # 旧版
        ('firing_rate',    ['firing_rate']),
        ('presence_ratio', ['presence_ratio']),
    ]

    metrics_to_request = []
    metric_col_map     = {}    # { 逻辑key : 实际输入列名 }

    for logic_key, candidates in METRIC_CANDIDATES:
        for candidate in candidates:
            if candidate in available_metrics:
                metrics_to_request.append(candidate)
                metric_col_map[logic_key] = candidate
                break
        else:
            print(f"          ⚠️  [{logic_key}] 所有候选名均不可用，跳过")

    # 去重
    metrics_to_request = list(dict.fromkeys(metrics_to_request))
    print(f"          实际请求: {metrics_to_request}")
    print(f"          列名映射: {metric_col_map}")

    analyzer.compute("quality_metrics", metric_names=metrics_to_request)
    print(f"  [2.5] ✅ 所有组件计算完成")

    # ── 2.6 导出到 Phy ────────────────────────────────────────────────────
    phy_dir = results_base / f'{region}_phy'
    if phy_dir.exists():
        shutil.rmtree(phy_dir)

    print(f"\n  [2.6] 🔄 导出 Phy 文件 → {phy_dir}")
    si.export_to_phy(
        analyzer,
        output_folder=str(phy_dir),
        compute_pc_features=False,
        compute_amplitudes=True
    )
    print(f"  [2.6] ✅ Phy 文件已生成")

    # ── 2.7 自动质控（动态 mask，列不存在自动跳过）───────────────────────
    qm = analyzer.get_extension("quality_metrics").get_data()
    print(f"\n  [2.7] 📊 {region} Quality Metrics 实际列名: {list(qm.columns)}")
    print(qm.to_string())

    mask_good = pd.Series(True, index=qm.index)

    _snr_col = metric_col_map.get('snr')
    if _snr_col and _snr_col in qm.columns:
        mask_good &= (qm[_snr_col] > THRESH_SNR)
        print(f"\n  📌 SNR       : '{_snr_col}' > {THRESH_SNR}")
    else:
        print(f"\n  ⚠️  SNR 列不存在，跳过")

    _isi_col = metric_col_map.get('isi')
    if _isi_col and _isi_col in qm.columns:
        mask_good &= (qm[_isi_col] < THRESH_ISI_RATIO)
        print(f"  📌 ISI       : '{_isi_col}' < {THRESH_ISI_RATIO}")
    else:
        print(f"  ⚠️  ISI 列不存在，跳过")

    _ns_col = metric_col_map.get('num_spikes')
    if _ns_col and _ns_col in qm.columns:
        mask_good &= (qm[_ns_col] > THRESH_MIN_SPIKES)
        print(f"  📌 n_spikes  : '{_ns_col}' > {THRESH_MIN_SPIKES}")
    else:
        print(f"  ⚠️  num_spikes 列不存在，跳过")

    _fr_col = metric_col_map.get('firing_rate')
    if _fr_col and _fr_col in qm.columns:
        mask_good &= (qm[_fr_col] > THRESH_MIN_FR)
        print(f"  📌 FR        : '{_fr_col}' > {THRESH_MIN_FR} Hz")
    else:
        print(f"  ⚠️  firing_rate 列不存在，跳过")

    _pr_col = metric_col_map.get('presence_ratio')
    if _pr_col and _pr_col in qm.columns:
        mask_good &= (qm[_pr_col] > THRESH_PRESENCE_RATIO)
        print(f"  📌 Presence  : '{_pr_col}' > {THRESH_PRESENCE_RATIO}")
    else:
        print(f"  ⚠️  presence_ratio 列不存在，跳过")

    good_unit_ids = qm.index[mask_good].tolist()

    # 逐单元打印筛选结果
    _show_cols = [c for c in [_snr_col, _isi_col, _ns_col, _fr_col, _pr_col]
                  if c and c in qm.columns]
    _col_w = 18
    header  = f"  {'Unit':<8}" + "".join(f"{c:>{_col_w}}" for c in _show_cols) + f"  {'结果':>8}"
    divider = f"  {'-' * (8 + _col_w * len(_show_cols) + 10)}"
    print(f"\n{header}\n{divider}")

    for uid in qm.index:
        row    = qm.loc[uid]
        marker = '✅ good' if uid in good_unit_ids else '❌'
        line   = f"  {str(uid):<8}"
        for c in _show_cols:
            val = row[c]
            line += f"{float(val):>{_col_w}.3f}" if not pd.isna(val) else f"{'NaN':>{_col_w}}"
        line += f"  {marker:>8}"
        print(line)

    print(f"\n  🏆 [{region}] 自动质控通过: {len(good_unit_ids)} / {n_units} 个单元")
    print(f"       通过的 unit IDs: {good_unit_ids}")

    # 把 good/noise 写回 Phy 的 cluster_group.tsv（人工复核时作为初始标记）
    cluster_group_path = phy_dir / 'cluster_group.tsv'
    if cluster_group_path.exists():
        cg_df = pd.read_csv(cluster_group_path, sep='\t')
        cg_df['group'] = 'noise'
        cg_df.loc[cg_df['cluster_id'].isin(good_unit_ids), 'group'] = 'good'
        cg_df.to_csv(cluster_group_path, sep='\t', index=False)
        print(f"  ✅ cluster_group.tsv 已更新（good/noise 写入 Phy）")

    # ── 2.8 提取并保存每个 unit 的 spike times（秒）─────────────────────
    spike_times_dir = results_base / f'{region}_spike_times'
    spike_times_dir.mkdir(exist_ok=True)

    # 用模板峰峰值确定每个 unit 的主通道
    templates_data = analyzer.get_extension("templates").get_data()
    # templates_data shape: (n_units, n_timepoints, n_channels)

    unit_ids = sorting.get_unit_ids()
    print(f"\n  [2.8] 💾 保存 spike times ({len(unit_ids)} 个单元) ...")

    for uid in unit_ids:
        # 样本点 → 秒
        spike_samples = sorting.get_unit_spike_train(unit_id=uid, segment_index=0)
        spike_times_s = spike_samples / SAMPLING_RATE

        # 保存单个 unit 的 .npy
        npy_path = spike_times_dir / f'{region}_unit{uid}_spikes_sec.npy'
        np.save(str(npy_path), spike_times_s)

        # 加入全局字典
        global_key                 = f'{region}_unit{uid}'
        all_spike_times[global_key] = spike_times_s

        # 找主通道（模板最大峰峰值所在通道）→ 转回 1-indexed 全局通道号
        try:
            uid_idx       = list(unit_ids).index(uid)
            template_unit = templates_data[uid_idx]              # (n_tp, n_ch)
            ptp_per_ch    = template_unit.max(axis=0) - template_unit.min(axis=0)
            best_ch_local  = int(np.argmax(ptp_per_ch))
            best_ch_global = ch_start + best_ch_local + 1        # 1-indexed
        except Exception:
            best_ch_global = -1

        # 取 quality metrics 行
        qm_row = qm.loc[uid] if uid in qm.index else pd.Series(dtype=float)

        all_units_meta.append({
            'region'              : region,
            'unit_id'             : uid,
            'global_key'          : global_key,
            'best_channel_1idx'   : best_ch_global,
            'n_spikes'            : len(spike_samples),
            'mean_fr_hz'          : round(len(spike_samples) / total_duration_s, 4),
            'snr'                 : round(qm_row.get(_snr_col, np.nan), 4) if (isinstance(qm_row, pd.Series) and _snr_col) else np.nan,
            'isi_violation'       : round(qm_row.get(_isi_col, np.nan), 4) if (isinstance(qm_row, pd.Series) and _isi_col) else np.nan,
            'firing_rate'         : round(qm_row.get(_fr_col,  np.nan), 4) if (isinstance(qm_row, pd.Series) and _fr_col)  else np.nan,
            'presence_ratio'      : round(qm_row.get(_pr_col,  np.nan), 4) if (isinstance(qm_row, pd.Series) and _pr_col)  else np.nan,
            'num_spikes_qm'       : int(qm_row.get(_ns_col,  -1))          if (isinstance(qm_row, pd.Series) and _ns_col)  else -1,
            'auto_label'          : 'good' if uid in good_unit_ids else 'noise',
        })

    print(f"  [2.8] ✅ 所有 spike times 已保存至: {spike_times_dir}")

# 主循环结束
print(f"\n{'='*65}")
print("🎉 BLOCK 2 主循环全部完成！")
print(f"{'='*65}")


# ██████████████████████████████████████████████████████████████████████████
# BLOCK 3 ── 汇总保存
# ██████████████████████████████████████████████████████████████████████████

print("\n" + "="*65)
print("BLOCK 3: 汇总保存所有结果")
print("="*65)

# ── 3.1 汇总元数据 CSV ───────────────────────────────────────────────────
summary_df  = pd.DataFrame(all_units_meta)
summary_csv = results_base / 'all_regions_units_summary.csv'
summary_df.to_csv(summary_csv, index=False)
print(f"\n📄 [3.1] 全量汇总表已保存: {summary_csv}")
print(summary_df.to_string())

# ── 3.2 全量 spike times（pickle）────────────────────────────────────────
# 格式: { 'ATL_unit1': np.array([t1,t2,...], dtype=float64) }  单位: 秒
all_pkl = results_base / 'all_spike_times.pkl'
with open(all_pkl, 'wb') as f:
    pickle.dump(all_spike_times, f, protocol=4)
print(f"\n📦 [3.2] 全量 spike times 已保存: {all_pkl}")
print(f"         共 {len(all_spike_times)} 个单元")

# ── 3.3 自动质控通过的 good units（pickle）───────────────────────────────
good_keys        = summary_df[summary_df['auto_label'] == 'good']['global_key'].tolist()
good_spike_times = {k: all_spike_times[k] for k in good_keys}
good_pkl         = results_base / 'good_units_spike_times.pkl'
with open(good_pkl, 'wb') as f:
    pickle.dump(good_spike_times, f, protocol=4)
print(f"\n✅ [3.3] 自动质控 good units 已保存: {good_pkl}")
print(f"         共 {len(good_spike_times)} 个单元")

# ── 3.4 各脑区统计汇总 ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print("📊 [3.4] 各脑区单元统计:")
print(f"{'='*65}")
region_summary = summary_df.groupby('region').agg(
    总单元数   = ('unit_id',    'count'),
    good单元数  = ('auto_label', lambda x: (x == 'good').sum()),
    noise单元数 = ('auto_label', lambda x: (x == 'noise').sum()),
    平均SNR    = ('snr',        'mean'),
    平均放电率  = ('firing_rate', 'mean'),
    总spike数   = ('n_spikes',   'sum'),
).round(3)
print(region_summary.to_string())

# ── 3.5 Phy 人工复核使用说明 ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("🔍 [3.5] Phy 人工复核命令（Sorting 完成后，打开终端执行）:")
print(f"{'='*65}")
for region in REGION_CHANNEL_MAP:
    phy_path = results_base / f'{region}_phy'
    print(f"\n  [{region}]")
    print(f"  cd '{phy_path}'")
    print(f"  phy template-gui params.py")
print("""
  标记规则:
    good  → 确认的单神经元 (single unit)
    mua   → 多单元混合 (multi-unit activity)
    noise → 噪声/伪影

  ⚠️  完成 Phy 标记后，运行 BLOCK 4 读取人工复核结果。
""")

print("="*65)
print("✅ BLOCK 3 完成！")
print("="*65)


# ██████████████████████████████████████████████████████████████████████████
# BLOCK 4 ── Phy 人工复核完成后运行
#            读取手动标记，生成最终 curated 数据集
# ⚠️  请在所有脑区 Phy 人工复核完成后，再单独运行此 Block
# ██████████████████████████████████████████████████████████████████████████

print("\n" + "="*65)
print("BLOCK 4: 读取 Phy 人工复核结果")
print("="*65)

# 重新加载全量 spike times（防止 kernel 重启后丢失）
with open(results_base / 'all_spike_times.pkl', 'rb') as f:
    all_spike_times = pickle.load(f)

# 重新加载汇总表（用于补充 metrics 信息）
summary_df = pd.read_csv(results_base / 'all_regions_units_summary.csv')

curated_spike_times = {}
curated_meta        = []

for region in REGION_CHANNEL_MAP:
    phy_dir = results_base / f'{region}_phy'
    cg_file = phy_dir / 'cluster_group.tsv'

    if not cg_file.exists():
        print(f"  ⚠️  [{region}] 找不到 cluster_group.tsv，跳过")
        continue

    cg_df    = pd.read_csv(cg_file, sep='\t')
    good_ids = cg_df[cg_df['group'] == 'good']['cluster_id'].tolist()
    mua_ids  = cg_df[cg_df['group'] == 'mua']['cluster_id'].tolist()
    print(f"  [{region}]  good: {good_ids}  |  mua: {mua_ids}")

    for uid in good_ids:
        key = f'{region}_unit{uid}'
        if key not in all_spike_times:
            print(f"    ⚠️  {key} 在 all_spike_times 中找不到，跳过")
            continue

        sp_times = all_spike_times[key]

        # 从汇总表里取 metrics
        row_mask = (summary_df['global_key'] == key)
        meta_row = summary_df[row_mask].iloc[0] if row_mask.any() else {}

        curated_spike_times[key] = sp_times
        curated_meta.append({
            'region'        : region,
            'unit_id'       : uid,
            'global_key'    : key,
            'n_spikes'      : len(sp_times),
            'mean_fr_hz'    : round(len(sp_times) / total_duration_s, 4),
            'snr'           : meta_row.get('snr',           np.nan) if isinstance(meta_row, pd.Series) else np.nan,
            'isi_violation' : meta_row.get('isi_violation', np.nan) if isinstance(meta_row, pd.Series) else np.nan,            
            'firing_rate'   : meta_row.get('firing_rate',   np.nan) if isinstance(meta_row, pd.Series) else np.nan,
            'presence_ratio': meta_row.get('presence_ratio',np.nan) if isinstance(meta_row, pd.Series) else np.nan,
            'phy_label'     : 'good',
        })

# 保存人工复核结果
curated_pkl = results_base / 'curated_spike_times.pkl'
with open(curated_pkl, 'wb') as f:
    pickle.dump(curated_spike_times, f, protocol=4)

curated_df  = pd.DataFrame(curated_meta)
curated_csv = results_base / 'curated_units_summary.csv'
curated_df.to_csv(curated_csv, index=False)

print(f"\n✅ 人工复核结果已保存!")
print(f"   spike times pkl : {curated_pkl}")
print(f"   汇总表 csv      : {curated_csv}")
print(f"   共 {len(curated_spike_times)} 个已确认的单神经元\n")
print(curated_df.to_string())

# 按脑区打印复核统计
print(f"\n{'='*65}")
print("📊 人工复核后各脑区单元数:")
print(f"{'='*65}")
if len(curated_df) > 0:
    print(curated_df.groupby('region')[['unit_id', 'mean_fr_hz', 'snr']].agg(
        单元数  = ('unit_id',    'count'),
        平均FR  = ('mean_fr_hz', 'mean'),
        平均SNR = ('snr',        'mean'),
    ).round(3).to_string())
else:
    print("  ⚠️  暂无 good 单元，请检查 Phy 标记是否正确。")

print("\n✅ BLOCK 4 完成！")


# ██████████████████████████████████████████████████████████████████████████
# BLOCK 5 ── Co-registration：与 Stimuli 对齐，计算 PSTH
# ⚠️  请将 stimuli_onsets 替换为你的真实刺激 onset 时间（单位：秒）
# ██████████████████████████████████████████████████████████████████████████

print("\n" + "="*65)
print("BLOCK 5: Co-registration & PSTH 计算")
print("="*65)

# ── 加载已确认单元 ────────────────────────────────────────────────────────
with open(results_base / 'curated_spike_times.pkl', 'rb') as f:
    curated_spike_times = pickle.load(f)
print(f"✅ 加载 {len(curated_spike_times)} 个已确认单元")

# ── ⚠️  替换为你的真实 stimuli onset 时间（单位：秒）─────────────────────
# 常见读取方式示例（三选一，取消注释即可）:
#
# 方式1: 从 .npy 文件读取
#   stimuli_onsets = np.load(base_folder / 'stimuli_onsets.npy')
#
# 方式2: 从 .csv 文件读取
#   stimuli_onsets = pd.read_csv(base_folder / 'events.csv')['onset_sec'].values
#
# 方式3: 从 .mat 文件读取
#   import h5py
#   with h5py.File(base_folder / 'stimuli.mat', 'r') as f:
#       stimuli_onsets = f['onset_times'][:].flatten()
#
# 当前为占位示例，请务必替换：
stimuli_onsets = np.array([10.0, 15.0, 20.0])   # ← ⚠️  替换为真实数据

print(f"✅ Stimuli onset 数量: {len(stimuli_onsets)}")
print(f"   时间范围: {stimuli_onsets.min():.2f}s – {stimuli_onsets.max():.2f}s")

# ── PSTH 参数（可按需调整）────────────────────────────────────────────────
PRE_STIM_S  = 0.5    # stimulus 前窗口 (秒)
POST_STIM_S = 1.0    # stimulus 后窗口 (秒)
BIN_SIZE_S  = 0.05   # bin 大小 (秒，50ms)

bins        = np.arange(-PRE_STIM_S, POST_STIM_S + BIN_SIZE_S, BIN_SIZE_S)
bin_centers = bins[:-1] + BIN_SIZE_S / 2
n_trials    = len(stimuli_onsets)
n_pre_bins  = int(PRE_STIM_S / BIN_SIZE_S)   # 基线 bin 数

print(f"\n  PSTH 参数:")
print(f"    Pre-stimulus  : {PRE_STIM_S*1000:.0f} ms")
print(f"    Post-stimulus : {POST_STIM_S*1000:.0f} ms")
print(f"    Bin size      : {BIN_SIZE_S*1000:.0f} ms")
print(f"    总 bin 数     : {len(bin_centers)}")

# ── 逐单元计算 PSTH ───────────────────────────────────────────────────────
psth_results = {}

print(f"\n  {'Unit':<25} {'基线FR(Hz)':>12} {'峰值FR(Hz)':>12} {'响应增益':>10}")
print(f"  {'-'*62}")

for unit_key, spike_times in curated_spike_times.items():
    trial_spikes = []

    for onset in stimuli_onsets:
        mask   = ((spike_times >= onset - PRE_STIM_S) &
                  (spike_times <= onset + POST_STIM_S))
        rel_st = spike_times[mask] - onset    # 相对时间（秒）
        trial_spikes.append(rel_st)

    # 合并所有 trial → histogram → 归一化为 firing rate (Hz)
    all_rel   = np.concatenate(trial_spikes) if trial_spikes else np.array([])
    counts, _ = np.histogram(all_rel, bins=bins)
    psth_hz   = counts / (n_trials * BIN_SIZE_S)

    # 计算基线（stimulus 前）和峰值（stimulus 后）
    baseline_fr = psth_hz[:n_pre_bins].mean()
    peak_fr     = psth_hz[n_pre_bins:].max()
    gain        = (peak_fr - baseline_fr) / (baseline_fr + 1e-6)   # 避免除零

    psth_results[unit_key] = {
        'bin_centers_s'        : bin_centers,
        'psth_hz'              : psth_hz,
        'baseline_fr_hz'       : baseline_fr,
        'peak_fr_hz'           : peak_fr,
        'response_gain'        : gain,
        'n_trials'             : n_trials,
        'raw_spikes_per_trial' : trial_spikes,
        'region'               : unit_key.split('_unit')[0],
    }

    print(f"  {unit_key:<25} {baseline_fr:>12.2f} {peak_fr:>12.2f} {gain:>10.2f}")

# 保存 PSTH 结果
psth_pkl = results_base / 'psth_results.pkl'
with open(psth_pkl, 'wb') as f:
    pickle.dump(psth_results, f, protocol=4)

print(f"\n✅ PSTH 结果已保存: {psth_pkl}")
print(f"   涵盖 {len(psth_results)} 个单元 × {n_trials} 个 trials")
print("\n✅ BLOCK 5 完成！")


# ██████████████████████████████████████████████████████████████████████████
# BLOCK 6 ── PSTH 可视化（按脑区分面板折线图）
# ██████████████████████████████████████████████████████████████████████████

print("\n" + "="*65)
print("BLOCK 6: PSTH 可视化")
print("="*65)

# 加载（防止 kernel 重启）
with open(results_base / 'psth_results.pkl', 'rb') as f:
    psth_results = pickle.load(f)

# 按脑区分组
region_units = {}
for unit_key, data in psth_results.items():
    r = data['region']
    region_units.setdefault(r, []).append(unit_key)

n_regions = len(region_units)

fig, axes = plt.subplots(
    n_regions, 1,
    figsize=(14, 4 * n_regions),
    squeeze=False
)

for row_idx, (region, unit_keys) in enumerate(region_units.items()):
    ax    = axes[row_idx][0]
    color = REGION_COLORS.get(region, '#607D8B')

    all_psth_mat = []
    for unit_key in unit_keys:
        d  = psth_results[unit_key]
        bc = d['bin_centers_s']
        hz = d['psth_hz']
        all_psth_mat.append(hz)
        ax.plot(bc, hz,
                alpha=0.35, linewidth=1.0,
                color=color, label=unit_key)

    # 均值曲线加粗
    if all_psth_mat:
        mean_psth = np.mean(all_psth_mat, axis=0)
        sem_psth  = np.std(all_psth_mat, axis=0) / np.sqrt(len(all_psth_mat))
        ax.plot(bc, mean_psth,
                color=color, linewidth=2.8,
                label=f'{region} Mean', zorder=5)
        # SEM 阴影
        ax.fill_between(bc,
                         mean_psth - sem_psth,
                         mean_psth + sem_psth,
                         alpha=0.15, color=color)

    # stimulus onset 竖线
    ax.axvline(x=0, color='black', linestyle='--',
               linewidth=1.5, alpha=0.85, label='Stimulus onset')
    # stimulus 时间窗阴影
    ax.axvspan(0, POST_STIM_S, alpha=0.04, color=color)
    # 基线窗阴影
    ax.axvspan(-PRE_STIM_S, 0, alpha=0.04, color='grey')

    ax.set_title(f'{region}   ({len(unit_keys)} units)',
                 fontsize=13, fontweight='bold', color=color, pad=8)
    ax.set_xlabel('Time relative to stimulus onset (s)', fontsize=11)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=11)
    ax.set_xlim(-PRE_STIM_S, POST_STIM_S)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='upper right',
              ncol=min(len(unit_keys) + 2, 6), framealpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle=':')

fig.suptitle('PSTH — All Brain Regions  (Mean ± SEM)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()

psth_pdf = results_base / 'psth_all_regions.pdf'
psth_png = results_base / 'psth_all_regions.png'
plt.savefig(psth_pdf, dpi=300, bbox_inches='tight')
plt.savefig(psth_png, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ PSTH 图已保存:\n   {psth_pdf}\n   {psth_png}")
print("\n✅ BLOCK 6 完成！")


# ██████████████████████████████████████████████████████████████████████████
# BLOCK 7 ── Raster Plot 可视化（每脑区独立一张，每 unit 一行）
# ██████████████████████████████████████████████████████████████████████████

print("\n" + "="*65)
print("BLOCK 7: Raster Plot 可视化")
print("="*65)

# 加载（防止 kernel 重启）
with open(results_base / 'psth_results.pkl', 'rb') as f:
    psth_results = pickle.load(f)

# 重新按脑区分组
region_units = {}
for unit_key, data in psth_results.items():
    r = data['region']
    region_units.setdefault(r, []).append(unit_key)

for region, unit_keys in region_units.items():
    n_units_r = len(unit_keys)
    if n_units_r == 0:
        continue

    color = REGION_COLORS.get(region, '#607D8B')

    # 每个 unit 占两行：上行 Raster，下行 PSTH bar
    n_rows = n_units_r * 2
    height_ratios = []
    for _ in range(n_units_r):
        height_ratios.extend([2, 1])    # raster 行高:psth 行高 = 2:1

    fig = plt.figure(figsize=(14, 3.5 * n_units_r))
    gs  = gridspec.GridSpec(
        n_rows, 1,
        height_ratios=height_ratios,
        hspace=0.08
    )

    for u_idx, unit_key in enumerate(unit_keys):
        data      = psth_results[unit_key]
        ax_raster = fig.add_subplot(gs[u_idx * 2])
        ax_psth   = fig.add_subplot(gs[u_idx * 2 + 1], sharex=ax_raster)

        # ── Raster ─────────────────────────────────────────────────────
        for trial_idx, trial_sp in enumerate(data['raw_spikes_per_trial']):
            ax_raster.vlines(
                trial_sp,
                trial_idx + 0.1,
                trial_idx + 0.9,
                color=color,
                linewidth=0.7,
                alpha=0.85
            )
        ax_raster.axvline(x=0, color='black', linestyle='--',
                          linewidth=1.2, alpha=0.8)
        ax_raster.set_xlim(-PRE_STIM_S, POST_STIM_S)
        ax_raster.set_ylim(0, data['n_trials'])
        ax_raster.set_ylabel(f"{unit_key}\n(trials)", fontsize=8)
        ax_raster.set_yticks([0, data['n_trials'] // 2, data['n_trials']])
        ax_raster.spines['top'].set_visible(False)
        ax_raster.spines['right'].set_visible(False)
        ax_raster.spines['bottom'].set_visible(False)
        plt.setp(ax_raster.get_xticklabels(), visible=False)

        # ── Mini PSTH（bar）──────────────────────────────────────────
        bc = data['bin_centers_s']
        hz = data['psth_hz']
        ax_psth.bar(bc, hz, width=BIN_SIZE_S * 0.9,
                    color=color, alpha=0.7, linewidth=0)
        ax_psth.axvline(x=0, color='black', linestyle='--',
                        linewidth=1.2, alpha=0.8)
        ax_psth.axhline(y=data['baseline_fr_hz'],
                        color='grey', linestyle=':', linewidth=1.0,
                        alpha=0.8, label=f"baseline {data['baseline_fr_hz']:.1f}Hz")
        ax_psth.set_xlim(-PRE_STIM_S, POST_STIM_S)
        ax_psth.set_ylim(bottom=0)
        ax_psth.set_ylabel('FR(Hz)', fontsize=7)
        ax_psth.spines['top'].set_visible(False)
        ax_psth.spines['right'].set_visible(False)
        ax_psth.tick_params(axis='both', labelsize=7)
        ax_psth.legend(fontsize=6, loc='upper right', framealpha=0.6)

        # 最后一个单元才显示 x 轴 label
        if u_idx == n_units_r - 1:
            ax_psth.set_xlabel('Time relative to stimulus onset (s)', fontsize=10)

    fig.suptitle(f'Raster + PSTH — {region}',
                 fontsize=13, fontweight='bold', color=color, y=1.005)

    raster_pdf = results_base / f'raster_{region}.pdf'
    raster_png = results_base / f'raster_{region}.png'
    plt.savefig(raster_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(raster_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ [{region}] Raster+PSTH 图已保存:\n   {raster_pdf}\n   {raster_png}")

print("\n✅ BLOCK 7 完成！")


# ██████████████████████████████████████████████████████████████████████████
# BLOCK 8 ── 最终输出文件结构总览 & 执行顺序提示
# ██████████████████████████████████████████████████████████████████████████

print(f"\n{'='*65}")
print("📁 BLOCK 8: 最终输出文件结构总览")
print(f"{'='*65}")

structure = f"""
{results_base}/
│
├── 📂 ATL_sorting/                 ← MountainSort4 原始输出
├── 📂 ATL_analyzer/                ← SortingAnalyzer (波形/模板/指标)
├── 📂 ATL_phy/                     ← Phy 可视化文件夹
│       └── cluster_group.tsv      ← good/noise 初始标记（可手动改）
├── 📂 ATL_spike_times/             ← ATL_unit1_spikes_sec.npy ...
│
├── 📂 HG_sorting/
├── 📂 HG_analyzer/
├── 📂 HG_phy/
├── 📂 HG_spike_times/
│
├── 📂 VMPFC_sorting/
├── 📂 VMPFC_analyzer/
├── 📂 VMPFC_phy/
├── 📂 VMPFC_spike_times/
│
├── 📂 Amygdala_sorting/
├── 📂 Amygdala_analyzer/
├── 📂 Amygdala_phy/
├── 📂 Amygdala_spike_times/
│
├── 📄 all_regions_units_summary.csv  ← 所有单元完整指标表（自动质控）
├── 📦 all_spike_times.pkl            ← 全量 spike times（秒）
├── 📦 good_units_spike_times.pkl     ← 自动质控通过的单元
│
├── 📄 curated_units_summary.csv      ← ⭐ Phy 人工复核后最终汇总表
├── 📦 curated_spike_times.pkl        ← ⭐ Phy 人工复核后最终 spike times
│
├── 📦 psth_results.pkl               ← PSTH 数据（bin_centers/psth_hz/...）
├── 🖼️  psth_all_regions.pdf/.png     ← 四脑区 PSTH 总览图
├── 🖼️  raster_ATL.pdf/.png           ← ATL Raster + mini PSTH
├── 🖼️  raster_HG.pdf/.png
├── 🖼️  raster_VMPFC.pdf/.png
└── 🖼️  raster_Amygdala.pdf/.png
"""
print(structure)

print(f"{'='*65}")
print("📋 推荐执行顺序:")
print(f"{'='*65}")
print("""
  1️⃣   BLOCK 0  →  环境初始化（每次启动 kernel 必跑）
  2️⃣   BLOCK 1  →  加载数据（每次启动 kernel 必跑）
  3️⃣   BLOCK 2  →  主循环 Sorting（耗时最长，跑一次即可）
  4️⃣   BLOCK 3  →  汇总保存（Sorting 完成后立即跑）
  ──────────────────────────────────────────────────────
  5️⃣   【终端】Phy 人工复核（four 个脑区逐一操作）
         cd '<results_base>/ATL_phy'
         phy template-gui params.py
         （其余脑区同理）
  ──────────────────────────────────────────────────────
  6️⃣   BLOCK 4  →  读取 Phy 人工复核结果
  7️⃣   BLOCK 5  →  替换 stimuli_onsets → Co-registration + PSTH
  8️⃣   BLOCK 6  →  PSTH 折线图可视化
  9️⃣   BLOCK 7  →  Raster + mini PSTH 可视化
  🔟   BLOCK 8  →  文件结构总览（随时可跑）

  💡  若只需重新出图，直接从 BLOCK 6/7 开始即可。
  💡  若 kernel 重启，至少重跑 BLOCK 0 + BLOCK 1，
      然后从需要的 Block 继续。
""")
print("="*65)
print("🎉 全部代码生成完毕！")
print("="*65)