# 来源、许可与改动范围

根 `LICENSE` 声明项目代码Apache-2.0；上游代码、模型、场景保留自身许可，不声明原创。

|部分|来源与依据|本项目使用方式|
|---|---|---|
|WPB Home场景|[6-robot/wpr_simulation2](https://github.com/6-robot/wpr_simulation2)，提交 `cd677ca41e6435550599311d8100ec4cf254eeab`；包内GPLv3 LICENSE|保留身体、家具布局与纹理；改为资源包，静态家具、精简装饰、整理RGB-D/TF，当前RGB-D水平视野调为90°|
|Ultralytics|[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)，8.4.202安装包LICENSE为AGPL-3.0|调用YOLOE预训练分割，无训练；独立Conda进程|
|模型资产|[ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0)|YOLOE与MobileCLIP文件；以各自上游条款为准，不由根LICENSE重新许可|
|CLIP依赖|[ultralytics/CLIP](https://github.com/ultralytics/CLIP)，提交 `7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40`；安装包AGPL-3.0|文本类别编码，依赖版本见锁定文件|
|ROS导航建图|[ROS2](https://github.com/ros2)、[Nav2](https://github.com/ros-navigation/navigation2)、[slam_toolbox](https://github.com/SteveMacenski/slam_toolbox)、[Gazebo ROS](https://github.com/ros-simulation/gazebo_ros_pkgs)|系统依赖，不打包其源码，保留组件许可|
|TF2局部修复|[geometry2 0.25.24](https://github.com/ros2/geometry2/tree/0.25.24)、[死锁修复记录](https://raw.githubusercontent.com/ros2/geometry2/0.25.24/tf2/CHANGELOG.rst)|从ROS apt下载校验tf2/tf2-ros并只在本项目加载；不修改系统，二进制不随代码包分发|
|云推理|[阿里云百炼](https://help.aliyun.com/zh/model-studio/)|HTTP调用千问，不含模型权重或API Key|

模型下载地址和 SHA256 统一维护在 [model_assets.json](../config/model_assets.json)。

参考：[VLFM](https://arxiv.org/abs/2312.03275)及官方[rai-opensource/vlfm](https://github.com/rai-opensource/vlfm)启发未知目标前沿搜索；《L-SNI: A Language-Driven Semantic Navigation System for Inspection Tasks》及ARTS_TEST启发持续对象语义图及对象周围停靠采样。当前未实现VLFM稠密价值图或L-SNI梯度评分；论文PDF不随代码包分发。

新增：状态机、常驻RGB-D体素对象图、在线搜索/停靠、ROS/Conda桥接、视觉优先运动、验证回复与封装。整体打包不改变第三方许可。
