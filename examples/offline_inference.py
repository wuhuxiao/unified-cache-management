import os
import time

os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def build_kv_transfer_config() -> KVTransferConfig | None:
    if os.getenv("ENABLE_UCM", "0") != "1":
        return None
    config_file = os.getenv(
        "UCM_CONFIG_FILE",
        "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml",
    )
    return KVTransferConfig(
        kv_connector="UCMConnector",
        kv_connector_module_path="ucm.integration.vllm.ucm_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"UCM_CONFIG_FILE": config_file},
    )


def build_llm(model: str) -> LLM:
    return LLM(
        model=model,
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "4")),
        data_parallel_size=int(os.getenv("DATA_PARALLEL_SIZE", "1")),
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
        max_num_batched_tokens=int(os.getenv("MAX_NUM_BATCHED_TOKENS", "700")),
        block_size=int(os.getenv("BLOCK_SIZE", "256")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
        enable_prefix_caching=os.getenv("ENABLE_PREFIX_CACHING", "1") != "0",
        kv_cache_dtype="fp8",
        kv_transfer_config=build_kv_transfer_config(),
        enable_expert_parallel=True,
        enable_flashinfer_autotune=False,
        disable_hybrid_kv_cache_manager=False,
        enforce_eager=os.getenv("ENFORCE_EAGER", "1") != "0",
    )


def print_output(
    llm: LLM,
    messages: list[dict[str, str]],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.chat([messages], sampling_params, use_tqdm=False)
    print("-" * 50)
    for output in outputs:
        completion = output.outputs[0]
        generated_text = completion.text
        input_tokens = len(output.prompt_token_ids or [])
        output_tokens = len(completion.token_ids or [])
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def main():
    model = os.getenv("MODEL_PATH", "/home/models/DeepSeek-V4-Flash")
    llm = build_llm(model)

    messages = [
        {"role": "system", "content": "You are a helpful text summarize assistant."},
        {"role": "user", "content": "你是什么模型"},
        {
            "role": "assistant",
            "content": "我是DeepSeek最新版本的模型，由深度求索公司创造。我是一个纯文本AI助手，擅长回答问题、提供信息、进行对话和协助各种文字处理任务。我的知识截止日期是2025年5月，支持1M的上下文长度，可以一次性处理像《三体》三部曲那样体量的书籍。\n\n有什么我可以帮你的吗？😊",
        },
        {"role": "user", "content": "介绍一下三体"},
        {
            "role": "user",
            "content": "《三体》是中国作家刘慈欣创作的一部硬科幻小说，也是“三体三部曲”（又名“地球往事”三部曲）的第一部。该系列包括《三体》、《三体Ⅱ：黑暗森林》和《三体Ⅲ：死神永生》。这部作品不仅在中国广受欢迎，还获得了国际认可，包括2015年的雨果奖最佳长篇小说奖，这是亚洲首次获得该奖项。\n\n### 故事梗概\n\n《三体》的故事始于文化大革命时期，当时天体物理学家叶文洁在绝望中向宇宙发送了信号，希望外星文明能够拯救地球。她的信号被距离地球约4光年的三体文明接收。三体文明位于一个拥有三颗太阳的星系，其环境极其恶劣，文明经历了无数次的毁灭与重生。三体人决定入侵地球，并开始监视和干扰地球的科学发展。\n\n与此同时，地球上出现了一个名为“三体游戏”的神秘虚拟现实游戏，玩家通过解谜逐渐了解到三体文明的存在。随着故事的发展，人类发现三体文明已经通过“智子”（一种微观粒子）锁死了地球的基础科学，使得人类无法在科技上超越三体人。最终，人类得知三体舰队正在向地球进发，预计400年后到达。\n\n### 核心主题\n\n1. **科学与文明**：小说探讨了科学进步与文明发展的关系，以及外星文明可能带来的威胁。\n2. **人性与道德**：在面临外星入侵的极端情况下，人类展现了自私、恐惧、希望和勇气等多重面貌。\n3. **宇宙社会学**：刘慈欣提出了“黑暗森林法则”，即宇宙中的文明之间由于无法信任彼此，往往会选择先发制人，导致宇宙成为一个充满危险的“黑暗森林”。\n4. **生存与毁灭**：小说深刻探讨了文明在宇宙中的生存困境，以及为了生存可能付出的代价。\n\n### 影响力\n\n《三体》不仅在中国引发了科幻热潮，还在全球范围内获得了大量读者和好评。它被翻译成多种语言，并被许多知名人士推荐，包括Facebook创始人马克·扎克伯格和美国前总统巴拉克·奥巴马。该系列也被改编为漫画、广播剧和电视剧，进一步扩大了其影响力。\n\n### 总结\n\n《三体》是一部兼具科学严谨性和哲学深度的科幻巨作，它通过宏大的宇宙视角和细腻的人类情感，引发了读者对文明、生存和宇宙的深刻思考。如果你对科幻、宇宙和人类命运感兴趣，这部作品绝对值得一读。",
        },
        {"role": "user", "content": "介绍一下刘慈欣"},
        {
            "role": "assistant",
            "content": "刘慈欣，1963年6月23日出生于中国山西省阳泉市，是中国当代最著名的科幻作家之一，被誉为“中国科幻文学的代表人物”和“中国科幻第一人”。他毕业于华北水利水电学院（现华北水利水电大学），曾长期在电厂担任计算机工程师，业余时间从事科幻小说创作。\n\n### 主要作品\n\n刘慈欣的代表作是“三体三部曲”（《三体》、《三体Ⅱ：黑暗森林》、《三体Ⅲ：死神永生》），该系列在全球范围内获得了极高的评价。此外，他还创作了许多中短篇科幻小说，如：\n\n- 《流浪地球》（后被改编为同名电影，票房大获成功）\n- 《乡村教师》\n- 《球状闪电》\n- 《朝闻道》\n- 《吞食者》\n- 《诗云》\n\n### 写作风格与特点\n\n1. **宏大的宇宙视野**：刘慈欣的作品常常将人类文明置于宇宙的宏大背景下，探讨文明在宇宙中的命运。\n2. **硬科幻基础**：他的作品基于扎实的科学理论，涉及物理学、天文学、宇宙学等多个领域，具有高度的科学严谨性。\n3. **哲学深度**：在科幻的外壳下，刘慈欣深入探讨了人性、道德、文明、生存等哲学命题。\n4. **中国元素**：他的作品融入了中国历史、文化和社会现实，如文化大革命、中国式集体主义等，使作品具有独特的东方视角。\n\n### 荣誉与影响\n\n- 2015年，《三体》获得雨果奖最佳长篇小说奖，这是亚洲首次获得该奖项。\n- 2017年，《三体Ⅲ：死神永生》获得轨迹奖最佳长篇小说奖。\n- 刘慈欣的作品被翻译成多种语言，在全球范围内拥有大量读者。\n- 他的作品被改编为电影、电视剧、广播剧、漫画等多种形式，进一步扩大了影响力。\n- 他被誉为“单枪匹马将中国科幻提升到世界水平”的作家。\n\n### 个人生活与评价\n\n刘慈欣为人低调，长期在电厂工作，直到《三体》获得巨大成功后，他才逐渐成为全职作家。他曾在采访中表示，自己的创作灵感来源于对科学的热爱和对人类命运的思考。他的作品不仅在中国引发了科幻热潮，也吸引了包括马克·扎克伯格、奥巴马等国际知名人士的关注。\n\n### 总结\n\n刘慈欣是一位兼具科学素养和人文关怀的科幻作家，他的作品以宏大的宇宙视角、严谨的科学基础和深刻的哲学思考，重新定义了华语科幻文学的高度。如果你对科幻感兴趣，刘慈欣的作品绝对不容错过。",
        },
        {"role": "user", "content": "介绍一下华北水利水电学院"},
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=16)

    print_output(llm, messages, sampling_params, "first")
    print_output(llm, messages, sampling_params, "second")


if __name__ == "__main__":
    main()
