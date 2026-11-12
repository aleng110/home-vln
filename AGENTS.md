# 项目维护约定

- 保留在线建图链路；任务端不读取 Gazebo 家具真值、先验地图或固定观察点。
- ROS 使用系统 Python；YOLO 使用独立 `yolo` 环境。
- 凭据仅在本机配置，不读取或输出值，不加入源码包；不隐式搜索旧项目。
- 保留 `/vln/cmd_vel_raw → guard → /cmd_vel`、动作取消、超时与带时间戳的 RGB-D/TF 校验。
- 语义几何是已见部分表面；准确区分实现内容与 VLFM/L-SNI 借鉴内容。
- 仿真验证使用独立 ROS 域，仅清理本次启动的进程；禁止 killall/pkill。
- 权重和生成数据存入忽略目录；实际验证结果写入 `docs/VALIDATION.md`。
