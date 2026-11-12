"""按已完成步骤生成用户回复，不编造检查结论。"""

NAMES = {
    'sofa': '沙发',
    'tea_table': '茶几',
    'dining_table': '餐桌',
    'chair': '椅子',
    'bed': '床',
    'bookshelf': '书柜',
    'cupboard': '橱柜',
}


def build_result(summary):
    outcome = summary.get('outcome', 'failed')
    completed = list(summary.get('completed_targets', []))
    target = NAMES.get(summary.get('current_target'), '当前目标')
    inspected = summary.get('inspection_answer')
    question = summary.get('inspection_question', '')
    names = [NAMES.get(label, label) for label in completed]
    if len(names) == 1:
        prefix = '我已到达' + names[0] + '附近。'
    elif names:
        prefix = (
            '我先到达了'
            + names[0]
            + '附近'
            + ''.join('，随后到达' + name + '附近' for name in names[1:])
            + '。'
        )
    else:
        prefix = ''
    error = str(summary.get('error', ''))
    if outcome == 'succeeded':
        if inspected:
            message = (prefix[:-1] + '，并完成检查：' if prefix else '已完成检查：') + str(inspected)
        else:
            message = prefix + ('自主探索任务已完成。' if not completed else '导航任务已完成。')
    elif outcome == 'cancelled':
        message = prefix + '任务已取消，我已停止移动。'
        if question and not inspected:
            message += '最终检查尚未完成。'
    else:
        if '不支持' in error or '支持列表' in error:
            reason = '指令包含当前不支持的目标或动作'
        elif '检查未完成' in error:
            reason = '未能完成指定表面的完整检查'
        elif any(x in error for x in ('全局规划服务', '等待 ROS action', '导航未就绪', '规划服务未就绪')):
            reason = '导航服务暂不可用，未能及时取得规划或运动结果'
        elif '避障点云' in error:
            reason = '避障感知数据暂不可用'
        elif '初始' in error and any(x in error for x in ('正前方', '前行', '向前')):
            reason = '起步通路暂不可用'
        elif '安全可达的接近' in error:
            reason = '已发现' + target + '，但暂时没有安全可达的接近路线'
        elif any(x in error for x in ('观察', '几何', '前沿', '候选', '决策', '停靠', '确认')):
            reason = '仍未能可靠确认' + target + '的位置或到达状态'
        elif any(x in error for x in ('云端', 'HTTP', 'Key', 'DASHSCOPE')):
            reason = '模型服务暂不可用'
        elif any(x in error for x in ('YOLO', '感知', '相机', 'RGB', 'TF')):
            reason = '视觉或定位数据暂不可用'
        else:
            reason = '任务遇到异常'
        message = prefix + '本次' + reason + '，我已停止移动。'
        if question and not inspected:
            message += '最终检查尚未完成，不能确认检查结果。'
        elif inspected:
            message += '检查结果：' + str(inspected)
    return dict(
        run_id=summary.get('run_id'),
        outcome=outcome,
        completed_targets=completed,
        current_target=summary.get('current_target'),
        inspection_question=question,
        inspection_performed=bool(inspected),
        inspection_answer=inspected,
        inspection_complete=bool(summary.get('inspection_complete', bool(inspected))),
        inspection_evidence=summary.get('inspection_evidence'),
        user_message=message,
        answer=inspected if outcome == 'succeeded' and inspected else message,
        error=summary.get('error'),
    )
