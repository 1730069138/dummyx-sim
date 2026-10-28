# GCC-VLA：按论文启动与复算实验

## 1. 本分支是什么

对应当前《GCC-VLA_小论文版.docx》，保留作者确认的旧路径。根目录 `README.md` 给出本分支克隆命令。本文的 F 通常指原固定方向、没有跳变限幅的完整 GCC-VLA；`F＋跳变限幅` 是另外的配置，不能将其200回合与原600回合中的 F 混为一个批次。

目录：

|位置|内容|
|---|---|
|`scripts/deploy/gcc_vla_paper/`|六方法统一入口、主实验配置、来源校验、VLSA重播代码|
|`scripts/deploy/`|实际几何、接触、观测器、感知及记录依赖|
|`models/gcc_vla_paper/`|原场景／X＋40 mm的冻结模型及初态；不是实验回合数据|
|`paper/gcc_vla/frozen/mechanism72/`|72回合运行、A/B开关、机理及8232状态诊断代码原样副本|
|`paper/gcc_vla/frozen/round2/`|40回合触限诊断、200回合方向对照、200回合F/B对照、5回合限幅探针的冻结代码|
|`paper/gcc_vla/frozen/round3/`|200回合限幅验证、统计检验与独立核验代码|
|`paper/gcc_vla/frozen/original600_analysis/`|原600载荷及耗时重算代码|
|`paper/gcc_vla/frozen/presubmission_analysis/`|原600统计检验、峰值时序和72回合诊断代码|
|`paper/gcc_vla/provenance/frozen_sources.json`|上述来源与SHA256；代码副本不改算法|
|`paper/gcc_vla/config/`|安装映射、带逐请求种子协议的策略服务启动模板|
|`outputs/`|初始化后安装的运行脚本、新机新生成结果；不提交到Git|

初始化只给第二轮调度器作一处有记录的基础设施适配：把读取旧600结果索引以排除旧种子的代码，改为相同的已知预留种子范围。原样调度器仍保存在 `frozen/round2/`，变更前后摘要写入初始化回执；未修改控制器、阈值、成功判断或统计方法。

## 2. 新电脑环境

要求 Linux、Python 3.10.20，路径如下。客户端主要依赖的实际版本保存在 `requirements-client.txt`，VLSA附加依赖在 `requirements-vlsa-extra.txt`。这是原机器的直接依赖快照，不是已经通过新机器完整安装测试的环境锁。

```bash
cd /home/jun/dummyx-sim
conda create --prefix /home/jun/anaconda3/envs/dummyx_vla python=3.10.20 -y
conda activate /home/jun/anaconda3/envs/dummyx_vla
pip install -r paper/gcc_vla/requirements-client.txt

# GroundingDINO：原电脑的源码提交，不下载模型进Git。
git clone https://github.com/IDEA-Research/GroundingDINO.git /home/jun/GroundingDINO
git -C /home/jun/GroundingDINO checkout 856dde20aee659246248e20734ef9ba5214f5e44
pip install -e /home/jun/GroundingDINO
```

GroundingDINO权重应放在 `/home/jun/GroundingDINO/weights/groundingdino_swint_ogc.pth`，配置为 `/home/jun/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`。请从旧电脑复制既有权重及BERT缓存，或按GroundingDINO原项目说明安装；本仓库不上传权重。旧实验在CPU上运行GroundingDINO，GPU主要用于远端策略服务。

VLSA使用独立环境，保持相同路径：

```bash
conda create --prefix /home/jun/anaconda3/envs/vlsa_env python=3.10.20 -y
conda activate /home/jun/anaconda3/envs/vlsa_env
pip install -r /home/jun/dummyx-sim/paper/gcc_vla/requirements-client.txt
pip install -r /home/jun/dummyx-sim/paper/gcc_vla/requirements-vlsa-extra.txt
pip install -e /home/jun/GroundingDINO
conda activate /home/jun/anaconda3/envs/dummyx_vla
```

无头渲染需要系统的OpenGL／OSMesa库；在Ubuntu中可安装 `libosmesa6 libegl1 libgl1`。Torch应与目标机器驱动匹配。代码使用MuJoCo 3.3.2，已提供的二进制模型按此版本加载。首次使用先测试：

```bash
python -c "import mujoco,numpy,scipy,torch,osqp; print(mujoco.__version__,numpy.__version__)"
python -c "from openpi_client import websocket_client_policy; from groundingdino.util.inference import load_model"
python paper/gcc_vla/manage.py verify
python paper/gcc_vla/manage.py prepare
```

`prepare` 逐项校验副本后，把脚本安装到下面各节的固定日期路径，创建空的audit/services/experiments目录，并生成统计计划摘要。不会创建成功结果或冒烟准入凭据，不会启动仿真或策略服务。遇到原有实验结果即停止。只运行一次；不要在有历史结果的电脑上执行。

72回合入口使用 `/tmp/gcc_revision_path`，初始化会写入正确代码目录。若重启后只丢失这个指针，可执行：

```bash
printf '%s' '/home/jun/dummyx-sim/outputs/gcc_vla_revision_20261028_132637' > /tmp/gcc_revision_path
```

## 3. π0策略服务：所有闭环实验的前提

本地客户端连接 `100.64.142.55`。六GPU／端口分别为0→8001、1→8002、2→8000、3→8003、4→8004、5→8005。远端需保留论文所用自定义OpenPI源码、环境、归一化文件和checkpoint：

`/home/jun/VLA/openpi/checkpoints/pi0_dummyx_lora/dummyx_lora_run0405/40000`

**这些远端源码与权重没有包含在本地代码仓库里。** 当前流程可让新客户端继续连接原远端。若连策略服务器也更换，先另行迁移原OpenPI环境／checkpoint；只安装官方OpenPI不等于恢复作者的自定义配置。服务检查会核对33个checkpoint文件、4个源码文件及逐请求种子协议。

SSH密钥不上传Git，按原路径在本机私下配置：

```bash
ssh-add /home/jun/ssh-keys/1
ssh -i /home/jun/ssh-keys/1 jun@100.64.142.55
```

服务必须包含 `fold_in(PRNGKey(episode_seed),request_index)` 包装器；直接运行普通serve_policy而不带包装器会被拒绝。模板由原始启动代码提取，保存在 `config/service_commands.json`。在六个终端分别执行：

```bash
# 不加--execute仅打印SSH命令，不连接服务。
python paper/gcc_vla/manage.py service --gpu 0
# 每个终端只启动对应GPU；将0依次替换为1、2、3、4、5。
python paper/gcc_vla/manage.py service --gpu 0 --execute
```

原有相同端口服务已在运行时，不要重复启动。完成实验后在各自终端Ctrl+C，确认只有自己的进程退出。不要通过killall清理共享服务器。

## 4. 论文—实验—代码对应表

|论文位置|实验集合及每组n|配置／种子|本说明命令|
|---|---|---|---|
|表1、表2、表3、表6|原600：6方法×2场景×50|纯π0、G、F、OSCBF、NEO、VLSA；610000／620000起|第5节|
|附录表D1|独立72：F/A/B×2×12|730000／740000起；跨GPU，不作严格配对|第6节|
|图3、表D3、表D5前8行|逐子步诊断40：4配置×2×5|810000／820000起|第7节exp1|
|表D2、表D6|方向对照200：fixed/normal×2×50|830000／840000起|第7节exp23中的exp2|
|表4、表D7|最终复核对照200：F/B×2×50|850000／860000起|第7节exp23中的exp3|
|表D3、表D5末行|限幅探针5：仅X＋40 mm|820000–820004，与exp1 F比较|第7节jump|
|表5、表D8|限幅验证200：F/limited×2×50|870000／880000起|第8节|
|附录D.3|8232个相关保存状态|来自原600的G、F；局部运动学反事实|第9节|
|附录表D4、接触检验|原600只读分析|峰值时序、非配对Fisher/MWU+Holm|第9节|

图1是方法流程图，图2是场景图，不是靠重复实验自动生成的统计结果。图3由第7节统计程序的 `figures/exp1_main_810000_F.png/.pdf` 生成。表号以当前小论文版为准；旧导表脚本的table7/tableD2文件名是历史命名，不能直接照旧编号插入当前论文。

## 5. 主对比：六方法×两场景×50＝600

```bash
cd /home/jun/dummyx-sim
conda activate /home/jun/anaconda3/envs/dummyx_vla

# 首先只预览，不连接服务、不推理。
python paper/gcc_vla/main600.py --method gcc-vla --scene main --episodes 50 --fixed-count

# 正式600：每一行循环调用都是独立配置，每配置每场景50回合。
for scene in main x40; do
  for method in pi0 geometry gcc-vla oscbf neo vlsa; do
    python paper/gcc_vla/main600.py --method "$method" --scene "$scene" \
      --episodes 50 --gpus 0,1,2,3,4,5 --fixed-count --execute || break 2
  done
done
```

`geometry`＝G，`gcc-vla`＝F且没有限幅；外部方法是论文适配实现。输出在 `outputs/gcc_vla_paper_runs/<时间戳>_<方法>_<场景>/`，每次新目录。默认种子为main610000–610049、x40620000–620049；新独立确认实验请在运行前统一另定 `--seed-start`，不把复用旧种子算作新的独立证据。

`main600.py`调用原统一入口，不改控制实现，只修正SSH认证文件路径；`--fixed-count`取消NEO原入口的“首批全部任务失败即停止”调度门槛，以保证预定50回合不按结果选样。技术异常仍停止、实际失败保留；改变的是调度停止规则并写入plan，非算法。省略该参数保留原入口行为。不要把NEO提前停止的6回合当成50回合。

新回合统计可用：

```bash
python paper/gcc_vla/summarize_runs.py \
  --run outputs/gcc_vla_paper_runs/实际本次目录 \
  --output outputs/gcc_vla_summary_新的目录名
# 可重复--run，同时统计多个方法/场景；同一身份重复会报错。
```

该脚本从原始终点和paper_truth重算，不根据目录success/fail猜测成功。表6耗时受硬件影响，不承诺在新电脑取得相同毫秒数。VLSA新运行若未保存统一1ms外载荷，则J/J4留空；不能用VLSA旧定义冲量代替统一指标。原论文VLSA载荷使用严格准入重播，见第9节，不能用其他批次补齐新运行缺失字段。

## 6. 72回合探索性消融：F/A/B

```bash
REV=/home/jun/dummyx-sim/outputs/gcc_vla_revision_20261028_132637
python paper/gcc_vla/manage.py check-services --batch mechanism72
python "$REV/scripts/gcc_schedule.py" --stage pilot
python paper/gcc_vla/manage.py admit72
python "$REV/scripts/gcc_schedule.py" --stage formal
python paper/gcc_vla/summarize_runs.py --run "$REV/experiments/formal" \
  --output outputs/gcc_vla_72_summary_新的目录名
```

pilot为6回合，不计正式72。新机准入检查源码身份、六配置覆盖、有效原始记录、A外力速度为零、B关闭子步复核；任务成功不是准入条件。它不是原稿中依赖已保存600状态的离线回归，原离线回归脚本另保留在frozen目录。F完整；A只关闭外力速度项，保留支撑反馈和高度抑制；B关闭整个1ms最终复核。72回合每组12，保留探索性描述和独立分母。

## 7. 逐子步诊断、方向对照、F/B和旧5回合探针

```bash
E=/home/jun/dummyx-sim/outputs/gcc_vla_experiments_round2_20261028_190400
python paper/gcc_vla/manage.py check-services --batch round2
python "$E/scripts/schedule_round2.py" --stage smoke
python "$E/scripts/audit_smoke_trace_readonly.py"
# 检查audit/smoke_admission.json的passed为true，再运行。
python "$E/scripts/schedule_round2.py" --stage exp1
python "$E/scripts/schedule_round2.py" --stage exp23
python "$E/scripts/schedule_round2.py" --stage jump
python "$E/scripts/analyze_round2.py" --root "$E" \
  --output-dir "$E/analysis_new_machine"
```

smoke有6回合，单独统计；exp1正式40；exp23正式400；jump额外5。**exp23一次同时完成方向对照和F/B对照**，不要重复运行当成两份结果。各配置同场景同seed使用同GPU，先后顺序预定交替。分析程序生成 `analysis_round2.json`、逐回合CSV、分组统计、检验、逐子步诊断表和图3的典型图。

限幅阈值直接使用论文冻结文件，不重新标定。不要加 `--freeze-jump-thresholds` 后把新标定值替换到200回合验证中。程序中的完整正式集合为440回合；5个探针和冒烟另计。输出目录必须新建；有技术中断按原身份记录，不能删除有效任务失败来补跑。

## 8. 200回合跳变限幅验证

```bash
R3=/home/jun/dummyx-sim/outputs/gcc_vla_round3_20261028_214716
# 第三轮调度使用离线BERT缓存。先在有网时下载到标准本地缓存：
HF_HOME=/home/jun/.cache/huggingface python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased')"
mkdir -p "$R3/audit/hf_cache/hub"
cp -a /home/jun/.cache/huggingface/hub/models--bert-base-uncased "$R3/audit/hf_cache/hub/"

python paper/gcc_vla/manage.py check-services --batch round3
python "$R3/scripts/schedule_round3.py" --stage smoke
python "$R3/scripts/admit_round3_smoke.py"
python "$R3/scripts/schedule_round3.py" --stage formal
python "$R3/scripts/analyze_round3_jump.py" \
  --output-dir "$R3/analysis_jump_final_v2" --require-complete
python "$R3/scripts/export_round3_tables.py" \
  --analysis "$R3/analysis_jump_final_v2/analysis_round3_jump.json" \
  --output-dir "$R3/task3/final_tables"
python "$R3/scripts/independent_verify_final200_readonly.py" \
  --primary-analysis "$R3/analysis_jump_final_v2/analysis_round3_jump.json"
```

BERT缓存是权重，不随Git上传；`cp`保留缓存相对软链接。若旧电脑已缓存，直接复制对应目录即可。原样调度器固定6路服务，不支持随意改成单GPU而声称相同协议。

F与F_limited各场景50，200正式回合；另有2个smoke。六阈值（rad）：0.031248799123169593、0.03577327030341545、0.03294372815894846、0.011915621510522133、0.03625579244210433、0.0005140508586996057。限幅位置在最终复核之后、写ctrl之前，首命令无前序目标不限幅。

统计保持预定双侧Fisher的6项Holm族，J/J4双侧MWU的4项Holm族，Cliff's delta方向limited−F。不足4秒的J4缺失不补零；触限、峰值、倾倒、时长描述统计。`tool_floor_contact`归为工具触地失败的修正已在冻结分析副本中，未修改原始事件。

200回合的导表文件仍叫table7/tableD2，这是上一稿导表命名；当前小论文应对应**表5、表D8**。不要将新机结果覆盖或冒充原论文这200回合。

## 9. 从既有数据重新计算论文结果（不重新做闭环）

本分支按作者要求不上传数据。只有从旧电脑另外恢复下列记录后才执行本节；缺少时应报告缺失，而不是把配置文件生成成历史结果。

|所需数据|旧电脑保留位置|
|---|---|
|原600完整canonical包|`outputs/pace_latest_evidence_20261027_1400/`|
|原72实际运行记录|`outputs/gcc_vla_revision_20261028_132637/experiments/`、`audit/`及恢复记录|
|第二轮440＋探针＋冒烟|`outputs/gcc_vla_experiments_round2_20261028_190400/experiments/`及冻结协议／身份|
|第三轮200＋冒烟|`outputs/gcc_vla_round3_20261028_214716/experiments/`及冻结协议／身份|
|旧600非配对检验和峰值审计中间件|`outputs/gcc_vla_round2_20261028_153421/audit/`|

**恢复已有数据时不要先prepare再复制覆盖结果。** 要么在空白克隆中使用本说明运行新实验，要么完整恢复旧数据包，再直接使用对应分析脚本；两条路线分开。原数据包已经包含冻结代码时优先按原包SHA使用它，Git中的原样副本用于缺项核对。

下列脚本是论文实际来源，默认写回固定审计位置，有已存在的结果时应先完整保留原包、在独立工作副本中运行。它们会拒绝覆盖或具有原固定输出名，不能在只读原始包中盲目执行：

```bash
# 8232局部反事实：只推进运动学，不执行物理步，不请求策略。
python outputs/gcc_vla_revision_20261028_132637/scripts/gcc_kinematic.py
# G/F控制机理、全程/固定窗口载荷、干预与最终复核统计。
python outputs/gcc_vla_revision_20261028_132637/scripts/gcc_audit.py
# 主论文接触顺应六项非配对检验（需原600审计输入）。
python outputs/gcc_vla_round2_20261028_153421/scripts/G与完整方法六项非配对检验.py
# 原600峰值相对夹爪闭合/双指接触的时序、72回合失败及阶段代理。
python outputs/gcc_vla_round2_20261028_153421/scripts/峰值接触与72失败只读诊断.py
# 表6计算耗时，读取原600逐回合日志。
python outputs/pace_latest_evidence_20261027_1400/载荷与耗时补全_20261027/timing_audit/recompute_timing.py
```

原600 VLSA统一载荷重播采用 `scripts/deploy/gcc_vla_paper/replay_metrics_20261027/replay_vlsa_exact.py`，需原canonical包的完整初始化、模型、对齐记录和逐回合索引。它是零推理开环重播，仍推进MuJoCo物理步，且必须通过严格轨迹／事件准入，不能叫纯静态读取：

```bash
# 将实际assignment_id填入；不是种子编号。只输出到新的目录。
/home/jun/anaconda3/envs/vlsa_env/bin/python \
  scripts/deploy/gcc_vla_paper/replay_metrics_20261027/replay_vlsa_exact.py \
  --output outputs/vlsa_replay_新的目录名 --ids 实际assignment_id
```

没有恢复原canonical包时，不执行此命令。`run_archived100.py`是历史剩余94条的恢复驱动，不适合当成新100回合的一键入口；代码保留溯源，但不要照它的固定断言运行全新批次。

## 10. 结果口径与验证范围

任务成功按现有放置判据，不等于严格完整入箱或放置后回位；STSR只覆盖碰柱、工具触地、柱体倾倒。J是受保护对象对外接触力模之和的积分，不是净矢量冲量；接触内部夹持力排除。等效外力、独立真实接触评价和位置目标日志不能混用。

CPU/GPU、环境或随机反馈变化可能导致新回合结果不同。同seed、同初态或首块相同均不意味着后续轨迹完全相同；不承诺重现每一项计数。实际任务失败完整保留；技术中断不算任务失败，先记明原因，再按原分配恢复，不增加有效样本数，不调整阈值。

本次发布只检查代码来源SHA、命令参数、固定分配、离线统计及模型资产加载；没有重新启动闭环、策略服务、训练或大规模仿真。新电脑的全套环境安装与闭环复跑需要在那里实际完成。代码分支不包含密钥、令牌或权重。
