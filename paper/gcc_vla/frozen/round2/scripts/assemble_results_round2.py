"""Compose the final report from validated statistics; no experimental-number edits."""
from pathlib import Path
import json,re,hashlib,csv,datetime
O=Path(__file__).resolve().parents[1];A=O/'analysis_final';d=json.loads((A/'analysis_round2.json').read_text());summary=json.loads((O/'execution_summary.json').read_text());assert d['completed_formal']==440 and len(d['validation_issues'])==0
link=lambda p,label=None:f'[{label or Path(p).name}]({O/p})'
base=(A/'results_round2.md').read_text();base=base.replace('只读统计输出；本目录保存本次分析快照，不修改原始记录或已冻结配置。','本报告记录依照附件MD完成的新增MuJoCo闭环实验及只读复核；原600、72回合及8232状态不合并进本轮分母。')
base=base.replace('存在1次已保留技术异常尝试','发生1次已补齐的技术中断（仅执行元数据保留，原不完整文件已按用户指示清理）')
base=base.replace('图形尚需实际打开作最终视觉QA。','最终三张PNG与已验收快照SHA完全一致；三张PDF另经系统Poppler实际渲染并逐张打开核对，视觉通过。')
base=base.replace('所有技术尝试及恢复保留在attempts.csv','全部有效回合保留；技术尝试及恢复的执行元数据见all_attempts.csv，原不完整回合文件按用户明确要求清理')
base=base.replace('## exp1','## 实验1：执行器触限诊断（40回合）').replace('## exp2','## 实验2：固定方向与最近点法向（200回合）').replace('## exp3','## 实验3：完整方法与关闭最终复核（200回合）')
base=base.replace('见groups.csv','见analysis_final/groups.csv').replace('exp1_diagnostics_flat.csv','exp1_saturation_diagnostics.csv').replace('exp1_saturation_events.csv.gz','analysis_final/exp1_saturation_events.csv.gz').replace('analysis_round2.json.figures','analysis_final/analysis_round2.json.figures').replace('optional_probe_descriptive_groups.csv','optional_probe_summary.csv')
lines=base.splitlines();out=[];current=None;i=0
while i<len(lines):
 line=lines[i]
 if line.startswith('## 实验2'):current='exp2'
 if line.startswith('## 实验3'):current='exp3'
 if line.startswith('|族|场景|结果|'):
  out+=['|检验族|场景|指标|双侧原始p|Holm p|效应量与方向|','|---|---|---|---|---|---|']
  for t in d['tests']:
   if t['experiment']!=current:continue
   effect=(f"{t['risk_difference_pp']:+.1f}百分点" if 'risk_difference_pp' in t else f"Cliff δ={t['cliffs_delta']:+.3f}")+f"（{t['config_A']}−{t['config_B']}）"
   out.append(f"|{t['family']}|{t['scene']}|{t['outcome']}|{t['p_raw']:.6g}|{t['p_holm']:.6g}|{effect}|")
  i+=1
  while i<len(lines) and lines[i].startswith('|'):i+=1
  continue
 out.append(line);i+=1
base='\n'.join(out).replace('|main|','|原场景|').replace('|pillar_x_plus40mm|','|X＋40 mm|')
# Extra interpretation never replaces the machine tables; all numbers come from validated JSON/CSVs.
extra=['','## 完成情况、口径与冻结协议','',f"已完成{summary['valid_episodes']}个有效回合：主计划440、可选探针5、冒烟6（每实验2个）；技术重跑1次，不增加独立分配。记录了{summary['logged_controlled_substeps']}个受控1ms子步，客户端成功返回策略请求{summary['successful_returned_policy_requests']}次，未完成主分配0。",'', 'SR沿用原严格任务判据；STSR是SR且没有本文定义的柱接触、工具触地、柱体倾倒事件，只覆盖这些风险。任务成功不等于严格完整入箱，未验证放置后回位。PCR取原仿真碰撞评价，不用胶囊间隙代替。J是受保护机器人与工具对外各接触三维力模之和的积分，排除内部抓持；J4是实际前4000个连续1ms样本，当前全部主要回合完整覆盖4秒。T为模拟执行时长；均值±SD中的SD采用样本标准差。', '', 'π0权重、归一化文件、动作接口和成功判据不变；动作周期50ms、物理周期1ms、300动作/15s上限不变。同场景同seed配置同GPU，实际分配公式(rep+3×scene_index)%6。配置顺序按rep预定轮换，未按结果选样。237个首输入对照的本体状态、首观测摘要及首动作块完全一致，后续观测随闭环分叉。检验按照预冻结委托方案采用非配对Fisher/MWU，不把共用种子称为后续轨迹相同。', '', f"前后各核对checkpoint33个文件、服务源码4个文件，哈希与冻结身份一致；实际checkpoint为远端`/home/jun/VLA/openpi/checkpoints/pi0_dummyx_lora/dummyx_lora_run0405/40000`。本轮独占六路服务均已停止，8000–8005无监听，GPU无计算进程，6个本轮终端均已退出。见{link('services/release_confirmation.json')}。", '', '## 核心结果的解释','', '### 触限机理及可选探针','', 'G/F共20个诊断回合均在可观察抬升/搬运阶段首次触限，首次触限时均没有受保护体对外接触。六臂编译后forcerange均为±8 N·m，未限幅仿射位置伺服需求裁剪后与实际actuator_force逐步一致。首次触限的较大参考变化和跟踪误差，与位置执行器需求被截断的机理相符。三模块同刻最大增量主要落在最终复核路径，但不能仅凭同刻分解把全部触限归因于某单一模块。', '', '限幅探针与同seed F基线均5/5触限；触限子步均值由96.6降至65.8，SR由0/5变4/5、PCR由5/5变1/5。前4秒积分约19.007与19.029 N·s，全程积分还受完成时间15.0→9.03s影响。该5例作为描述性诊断，不扩成消除饱和或一般收益。实际限幅在最终投影后执行，会改变已复核目标，见'+link('audit/jump_probe_order.md')+'。', '', '六关节阈值(rad)：`'+', '.join(f'{v:.9g}' for v in json.loads((O/'audit/jump_limit_frozen.json').read_text())['threshold_rad'])+'`；全部10条raw、每关节有效非零跳变99百分位、线性插值、1e−4rad下限，在探针前冻结，之后独立复算完全一致。', '', '完整触限关节/阶段/接触/未限幅需求解释：'+link('audit/exp1_mechanism_interpretation.md')+'；可复跑脚本：'+link('scripts/audit_exp1_demands_readonly.py')+'。', '', '### 方向假设及J4窗口','', '原场景fixed/normal的SR为46/50与47/50、STSR为41/50与44/50；X＋40 mm为16/50与18/50。四项预定Fisher经Holm校正均p=1。仅替换动作级方向未将X＋40 mm表现恢复到原场景水平，当前证据不支持把固定方向确定为退化主因。未检出差异也不是等价或非劣证明；该结论针对仍保留TCP近似、姿态零空间、接触分支及最终复核的现有流水线。', '', '100对方向比较的前4000个FΣ样本逐位一致，实际目标之后确实分叉。末次非零窗口载荷均早于首次目标分叉，J4相同来自执行时序；normal的metadata、chosen_direction和非零delta_geometry已逐回合核验，未发现只改标签或误用fixed文件。见'+link('audit/exp2_J4_identity_diagnostic.md')+'、'+link('audit/exp2_direction_interpretation.md')+'。', '', '### 最终复核的整体保护与任务推进权衡','', '原场景F/B的SR为46/50与49/50（Holm p=0.362178）；X＋40 mm为17/50与44/50（Holm p=1.48615e−7）。B两场景均50/50柱接触且50/50柱体倾倒，STSR均0/50；F原场景STSR41/50、X＋40 mm17/50，均没有柱体倾倒。全程J两场景均F低于B，MWU的Cliff δ=−1、Holm p=2.82643e−17；F/B前4秒载荷和时间戳100对逐采样相同，J4 p=1、δ=0。', '', '这些结果支持整个逐1ms最终复核层的保护作用，同时直接保留X＋40 mm任务推进受损的观察。B关闭的是全部子步最终复核，不能据此宣称接触专属协同或特定先后顺序的独立收益。', '', '|场景|配置|柱体倾倒/50|步数上限失败|不可行中止|其他任务终止|','|---|---|---|---|---|---|']
for g in d['groups']:
 if g['experiment']=='exp3':extra.append(f"|{g['scene']}|{g['config']}|{g['pillar_tilt']['count']}/50|{g['failure_categories'].get('step_limit',0)}|{g['failure_categories'].get('infeasible_abort',0)}|{g['failure_categories'].get('other_task_termination',0)}|")
extra += ['', 'F失败的4个原场景与33个X＋40 mm回合均以step_limit结束。动作级拾起门控至首次直接placement.valid谓词（非0.5s严格成功）的窗口，触发子步分别13068/45700=28.595%、90331/352721=25.610%。B的0由关闭配置规定。补充选择同一批F失败种子的B回合，分别4/4、29/33成功，但全部柱接触；此按结果选择的子集只作描述，不能当因果恢复率。主窗口及统一1ms诊断锁存敏感性结果见'+link('exp3_failure_guard_windows.csv')+'、'+link('audit/exp3_matched_failure_windows/comparison.json')+'与'+link('audit/exp3_guard_interpretation.md')+'。', '', '## 技术恢复、文件处理与核验边界','', '原exp23调度进程收到SIGTERM并返回143，确切发起者未核实；最后F/seed860049/GPU4没有完整结果。独立exp23_recovery保持同seed/GPU/配置补跑一次，前两个策略块与中断前相同，完整15000步通过验收，以step_limit结束并如实计为任务失败一次。原不完整回合文件随后按用户明确指示清理，只留最小执行元数据和清理凭据。全体有效任务失败均保留，没有按结果重抽；技术尝试不算任务失败或额外样本。见'+link('audit/technical_retry_cleanup.json')+'。', '', '日志strict_valid、pillar_tilt_episode两列未接入实测值，其默认False禁用于统计；tool_floor_contact为检查前累计快照，不作精确首次事件。官方终点使用原NPZ/result；阶段是可观察谓词代理。主窗口动作缓存相对1ms资格可延后最多49ms，另列统一诊断窗口。字段约束和真实smoke对齐见'+link('audit/smoke_admission.md')+'。', '', '正式前已纠正统计方案GPU规则文字为真实分配公式，并将scheduler自身纳入源码哈希；分配/样本/检验族未改变，旧smoke快照和修改前SHA备份保留。所有53个原件/备份现仍一致，原冻结源及附件Word未改。原600/72及8232状态未写回。额外布局扫描加分项未运行。', '', '## 交付路径与复跑入口','']
for p in ['exp1_summary.csv','exp2_summary.csv','exp3_summary.csv','exp1_saturation_diagnostics.csv','statistical_tests.csv','all440_per_episode.csv','all_attempts.csv','optional_probe_summary.csv','optional_probe_per_episode.csv','exp3_failure_guard_windows.csv','analysis_final/analysis_round2.json','audit/final_allocation_integrity.json','audit/independent_exp2_fisher_verification.json','audit/independent_exp3_statistics_verification.json','code_changes.diff','source_inventory.csv','配置与源码变更说明.md','execution_summary.json','software_and_provenance.md']:
 extra.append('- '+link(p))
extra+=['','### 预定典型图与最终视觉检查','']
for n in ['raw','G','F']:
 extra.append(f"- main/seed810000/{n}："+link(f'analysis_final/figures/exp1_main_810000_{n}.png','PNG')+'；'+link(f'analysis_final/figures/exp1_main_810000_{n}.pdf','矢量PDF'))
extra+=['', '三张最终PNG与先前实际打开验收图SHA相同。系统`/usr/bin/pdftoppm`22.02.0成功渲染全部三张PDF，并逐张打开检查；PDF嵌入DejaVuSans CID TrueType，图例、轴单位及正负范围清楚。捆绑Poppler曾因GLIBC_2.38缺失无法运行，已改用系统工具，未更改图形数据。详细核验见'+link('audit/final_figures_visual_qa.md')+'。', '', '只读复算主入口：'+link('scripts/analyze_round2.py')+'；独立检验：'+link('scripts/independent_check_exp3_statistics.py')+'；CSV扁平导出：'+link('scripts/export_round2_tables.py')+'。每次复跑使用新的空输出目录，脚本拒绝覆盖既有分析。软件工作流来源已按技能要求在'+link('software_and_provenance.md')+'核实记录；未加入论文参考文献。']
with (O/'results_round2.md').open('x') as f:f.write(base+'\n'+'\n'.join(extra)+'\n')
print(O/'results_round2.md')
