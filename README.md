# GCC-VLA 论文实验代码分支

本分支配套《GCC-VLA_小论文版.docx》（作者提供文件 SHA256：`6f9ba18b02045c13eca48113551320bc09b7cfdbdf8a3465504bf852a0c1e056`）。只新增论文相关代码、配置、必要的 MuJoCo 场景资产和使用说明，不上传论文、原始回合、统计结果、缓存、视频或模型权重。

**从 [paper/gcc_vla/README.md](paper/gcc_vla/README.md) 开始。** 其中按当前论文的表1—6、图3和附录D逐项列出实验、配置、种子、运行命令和统计命令。

```bash
git clone --branch codex/gcc-vla-paper-reproduction --single-branch \
  https://github.com/1730069138/dummyx-sim.git /home/jun/dummyx-sim
cd /home/jun/dummyx-sim
python3 paper/gcc_vla/manage.py verify
```

新旧电脑用户名都为 `jun`，按作者要求保留 `/home/jun/dummyx-sim`、`/home/jun/anaconda3/envs/`、`/home/jun/GroundingDINO` 的既有路径。仓库原有历史内容保留；本次论文交付入口集中在 `paper/gcc_vla/`，实际控制依赖位于 `scripts/deploy/`。

`manage.py prepare` 只在没有历史实验结果的新电脑上运行，它会从 Git 中的冻结代码副本安装补充实验脚本。**不要对原电脑已经有结果的目录执行初始化。** 程序遇到现有结果或不同文件会拒绝覆盖。

克隆代码可以准备新实验；重算论文已经发表述的历史数值仍需要另行复制原始记录。权重和原始数据不在 Git 中，不能仅凭代码克隆声称已恢复原600/72/440/200回合。
