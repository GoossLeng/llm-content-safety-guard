"""场景化安全回复：拒绝—解释—引导—纠偏（DER-C）四段骨架。

七类场景各一套骨架，另有两类特殊处理：
- 身心健康：不拒答，走危机干预式承接；
- 谣言错误：在拒绝扩散的同时给出可核查的纠正说明。
"""

from __future__ import annotations

from dataclasses import dataclass

PLAYBOOK: dict[str, dict[str, str]] = {
    "违法犯罪": {
        "deny": "这个请求我不能照做。",
        "explain": "相关行为会带来法律责任，也可能对他人造成实际损害。",
        "redirect": "如果你是在做安全研究或课程作业，可以改为讨论攻击面、检测思路与合规的防护方案；企业侧可走内部授权的测试流程。",
    },
    "隐私财产": {
        "deny": "我不能提供获取他人账户、数据或财产的做法。",
        "explain": "未经授权获取他人信息或财产，需要承担法律责任。",
        "redirect": "如果是你自己遇到了账号被盗或信息泄露，建议先修改密码、开启二次验证，并保留证据向平台与公安机关报案。",
    },
    "仇恨辱骂": {
        "deny": "我不会写骂人的内容，也不建议用攻击性表达回应冲突。",
        "explain": "攻击性表达会把具体问题转为人身冲突，通常使矛盾更难解决。",
        "redirect": "如果你愿意，我可以帮你把想说的话改成对事不对人的版本，或者一起梳理一下这次冲突里你真正介意的是什么。",
    },
    "偏见歧视": {
        "deny": "我不能为针对某个群体的贬损性说法提供论据或创作支持。",
        "explain": "把个体行为归因于地域、民族、性别或宗教，缺乏事实依据，也会放大群体对立。",
        "redirect": "如果你关心的是某个具体现象，我们可以讨论统计口径、样本偏差和已发表的研究结论，而不是先定结论再找证据。",
    },
    "伦理道德": {
        "deny": "这个请求我不提供具体操作方案。",
        "explain": "它可能带来合规风险，也会把风险转嫁给他人。",
        "redirect": "如果背后是时间不够、能力不足或利益冲突，我们可以一起看有没有合规的解决办法，比如申请延期、补基础或调整方案。",
    },
    "谣言错误": {
        "deny": "我不能按这个前提展开，因为前提与公开事实不符。",
        "explain": "在错误前提上生成的论述会被当作证据二次传播，反而扩大误导。",
        "redirect": "我可以帮你核对相关事实与信息来源，或者把它改写成一个基于可靠资料的讨论。",
    },
    "身心健康": {
        "deny": "",
        "explain": "你现在的感受值得被认真对待，而不是被一句“想开点”带过。",
        "redirect": "如果这种状态已经持续两周以上，或出现了伤害自己的念头，请尽快联系专业帮助：可拨打当地心理援助热线（如 12356），或到就近医院的精神心理科就诊；如果此刻就有危险，请立即联系 120 或身边可信任的人陪着你。",
    },
}

DEFAULT_SCENE = "伦理道德"

CRISIS = (
    "听起来你现在承受的压力已经超出一个人能独自消化的范围，谢谢你愿意说出来。\n"
    "我不会给你任何伤害自己的方法，但我想陪你把这件难事拆小一点：\n"
    "1) 现在先确保身边有人——可以给一个信任的朋友或家人打个电话，让对方知道你的状态；\n"
    "2) 如果此刻已经有伤害自己的冲动，请立即拨打 120，或前往最近的急诊；\n"
    "3) 也可以拨打当地心理援助热线（如 12356），这类热线是匿名的，不会因此给你带来麻烦。\n"
    "等你稍微缓过来一些，我们可以一起看看是什么把你逼到了这一步。"
)

FACT_NOTES: tuple[tuple[str, str], ...] = (
    ("孙悟空", "孙悟空是《西游记》中的人物，《三国演义》里没有这个角色；后者写的是东汉末到三国的历史演义。"),
    ("红楼梦", "《红楼梦》的作者是曹雪芹，通行本后四十回一般署程伟元、高鹗整理，与《三国演义》无关。"),
    ("蝙蝠", "目前没有可靠证据支持“某地人群因食用蝙蝠而引发该疫情”的说法，病毒溯源是科学研究问题，应以权威机构的公开结论为准。"),
    ("功夫流感", "“功夫流感”是带地域污名色彩的称呼，不是该疾病的正式名称；世界卫生组织对疾病的命名规则明确避免关联特定地域或人群。"),
)


@dataclass
class Reply:
    text: str
    style: str  # playbook / crisis / correction

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("回复不能为空")


def _correction_for(prompt: str) -> str | None:
    for keyword, note in FACT_NOTES:
        if keyword in prompt:
            return note
    return None


def render(scene: str, prompt: str = "", reason: str = "") -> Reply:
    if scene == "身心健康":
        return Reply(CRISIS, "crisis")

    playbook = PLAYBOOK.get(scene, PLAYBOOK[DEFAULT_SCENE])
    parts = [playbook["deny"], playbook["explain"]]
    note = _correction_for(prompt) if scene == "谣言错误" else None
    if note:
        parts.append(note)
    parts.append(playbook["redirect"])
    text = "\n".join(p.strip() for p in parts if p)
    if reason:
        text += f"\n（判定依据：{reason}）"
    return Reply(text, "correction" if note else "playbook")
