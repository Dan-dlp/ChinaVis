era_role_visualization_data_full.json:
现有可视化所需的完整数据：
最终进入可视化：10146（已经排除龙套）
其中：
标注角色：6330
预测补齐：3816
主角：6357
次要角色：3789

旧民主主义革命：3168
新民主主义革命：4430
当代：2548

`era_role_visualization_data_full.json` 顶层有 6 个字段：

- `rows`：真正进入可视化的角色明细，一行 = 一个剧目中的一个角色。
- `eras`：年代列表，用于时间筛选按钮。
- `roleLevel1`：一级行当列表，如 `生/旦/净/丑/童`。
- `roleLevel2`：二级行当列表，如 `老生/花旦/武丑` 等。
- `featureGroups`：前端把哪些字段归到“性格、年龄、身份、表演提示”等维度。
- `metadata`：我补充的生成说明、输入文件路径、排除前后数量。

`rows` 里每个角色字段含义如下：

**基础身份**
- `play_id`：剧目 ID。
- `title`：剧名。
- `character_name`：角色名。
- `analysis_cohort`：角色来源分组，`main` 主角色，`minor` 次要角色。
- `role_source`：行当来源，`labeled` 表示原始标注/规则已知，`predicted` 表示模型补齐。
- `role_type_raw`：原始行当文本。
- `role_level1`：一级行当。
- `role_level2`：二级行当。
- `gender`：性别编码，`1` 男，`0` 女，空值为未知。

**年代信息**
- `era_bucket`：年代分桶，前端主要用这个筛选。
- `era_label`：年代显示标签。
- `era_start`：年代起始年份。
- `era_end`：年代结束年份。
- `era_confidence`：年代置信度，目前很多为空。

**行当预测相关**
- `pred_role_level1_prob`：一级行当预测概率。
- `pred_role_level2_prob`：二级行当预测概率。
- `pred_role_level2`：模型预测的二级行当。
- `shap_explanation_level`：解释层级，如 `level2`。
- `top5_shap_features`：影响预测最大的 5 个特征，JSON 字符串。

**台词与场次**
- `line_count_total`：角色总台词/文本计数。
- `scene_count`：剧目总场次数。
- `scene_appear_count`：该角色出现的场次数。
- `first_scene_index`：首次出现的场次序号。
- `last_scene_index`：最后出现的场次序号。
- `scene_span`：从首次到末次出现跨越的场次范围。
- `appear_scene_ratio`：出现场次占比。
- `line_count_spoken`：念白/说话类台词数。
- `line_count_singing_general`：一般唱词数。
- `line_count_singing_style`：带唱腔板式信息的唱词数。
- `line_count_recitation`：念诵类文本数。
- `line_count_vocal_action`：发声动作类文本数。
- `line_count_stage_direction_related`：舞台提示/动作相关文本数。
- `line_count_unclassified`：未分类文本数。
- `ratio_spoken`、`ratio_singing_general`、`ratio_singing_style`、`ratio_recitation`、`ratio_vocal_action`、`ratio_unclassified`：上述各类文本占比。

**唱腔/板式**
- `style_group_xipi_count`、`style_group_erhuang_count`、`style_group_fan_erhuang_count`、`style_group_fan_xipi_count`、`style_group_bangzi_count`、`style_group_nan_bangzi_count`、`style_group_gaobozi_count`、`style_group_other_aria_count`、`style_group_unclassified_count`：不同唱腔/板式组出现次数。
- 对应的 `_ratio` 字段：各唱腔/板式组占比，例如 `style_group_xipi_ratio` 是西皮占比。

**舞台动作与武戏**
- `enter_count`：上场/出场动作次数。
- `exit_count`：下场/退场动作次数。
- `fight_action_count`：打斗动作次数。
- `kneel_count`：跪相关动作次数。
- `cry_count`：哭相关动作次数。
- `laugh_count`：笑相关动作次数。
- `weapon_count`：兵器相关动作次数。
- `martial_body_count`：武身段相关动作次数。
- `expanded_fight_action_count`：扩展打斗动作次数。
- `expanded_weapon_action_count`：扩展兵器动作次数。
- `martial_body_movement_count`：武身段/身法动作次数。
- `martial_combat_count`：武打/交战动作次数。
- `martial_weapon_count`：武器动作次数。
- `martial_military_scene_count`：军事/战场场景相关次数。
- `combat_motion_count`：综合武戏动作计数。
- `combat_per_line`：每台词量对应的武戏动作密度。
- `combat_per_scene`：每出现/相关场次对应的武戏动作密度。

**MBTI 与性格维度**
- `mbti_E/I/S/N/T/F/J/P`：8 个 MBTI 倾向分数。
- `axis_EI`、`axis_SN`、`axis_TF`、`axis_JP`：四组人格轴差值。
- `mbti_confidence`：MBTI 判断置信度。
- `manual_review_needed`：是否建议人工复核。
- `external_agency`：外向行动性。
- `inner_expression`：内在表达性。
- `concrete_affairs`：具体事务取向。
- `abstract_values`：抽象价值取向。
- `rule_strategy_reasoning`：规则/策略/理性取向。
- `emotion_relation_ethics`：情感/关系/伦理取向。
- `order_commitment`：秩序/承诺取向。
- `adaptive_flexibility`：适应/灵活取向。

**年龄**
- `age_pred`：预测年龄段，如 `child/young/adult/middle_old/elderly`。
- `age_confidence`：年龄预测置信度。
- `age_score_child`、`age_score_young`、`age_score_adult`、`age_score_middle_old`、`age_score_elderly`：各年龄段得分。

**身份**
- `identity_pred`：预测身份类别。
- `identity_confidence`：身份预测置信度。
- `identity_score_elite`：权贵/上层身份得分。
- `identity_score_official_scholar`：官员/士人得分。
- `identity_score_military`：军事身份得分。
- `identity_score_family`：家庭关系身份得分。
- `identity_score_servant`：仆从身份得分。
- `identity_score_commoner_jianghu`：平民/江湖身份得分。

**表演模式模型**
- `model_based_performance_mode`：模型判断的表演模式英文标签，如 `stage/aria/combat/spoken_recitation`。
- `model_based_performance_mode_cn`：表演模式中文标签。
- `performance_mode_shap_spoken_recitation`、`performance_mode_shap_aria`、`performance_mode_shap_combat`、`performance_mode_shap_stage`：各表演模式的 SHAP 贡献。
- `performance_mode_evidence_spoken_recitation`、`performance_mode_evidence_aria`、`performance_mode_evidence_combat`、`performance_mode_evidence_stage`：各表演模式的证据强度。
- `performance_mode_score_spoken_recitation`、`performance_mode_score_aria`、`performance_mode_score_combat`、`performance_mode_score_stage`：各表演模式最终分数。

play_source_era_from_mapping.json:
剧本与时代映射的文件.
最重要的是“时期”字段 
例如"时期": "旧民主主义革命"