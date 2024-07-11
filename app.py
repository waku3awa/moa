"""
Japanese Mixture of Agents

based on https://github.com/togethercomputer/MoA/tree/main

        Code by shi3z(2024/7/5)
"""
# from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
# import openai
import requests
from rich import print
from rich.console import Console
from rich.markdown import Markdown
# from rich.prompt import Prompt
import copy
import multiprocessing as mp
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

console = Console()

models = [
    ["Mistral/mixtral-8x7b-instruct-v0.1/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", "172.19.128.1:1234"],
    ["mradermacher/Llama-3-ELYZA-JP-8B-GGUF/Llama-3-ELYZA-JP-8B.Q8_0.gguf", "172.19.128.1:1234"],
    ["Xwin/xwin-lm-13b-v0.1/xwin-lm-13b-v0.1.Q5_K_S.gguf", "172.19.128.1:1234"]
]
models_reference = ["mradermacher/Llama-3-ELYZA-JP-8B-GGUF/Llama-3-ELYZA-JP-8B.Q8_0.gguf", "172.19.128.1:1234"]

welcome_message = f"""
# Welcome to the Free AI MoA (Mixture-of-Agents)  demo!

Mixture of Agents (MoA) is a novel approach that leverages the collective strengths of multiple LLMs to enhance performance, achieving state-of-the-art results. By employing a layered architecture where each layer comprises several LLM agents, MoA significantly outperforms GPT-4 Omni’s 57.5% on AlpacaEval 2.0 with a score of 65.1%, using only open-source models!

This demo uses the following LLMs as reference models, then passes the results to the aggregate model for the final response:
- {models_reference[0]}
"""


def generate_direct(
        messages,
        model,
        max_tokens,
        temperature,
    ):

    output = model(messages)

    return output.text


def generate_with_local_api(
        messages,
        model,
        max_tokens,
        temperature,
    ):
    output = None

    modelpath, port = model
    res = requests.post(
        f"http://{port}/v1/chat/completions",
        json={
            "model": modelpath,
            "max_tokens": max_tokens,
            "temperature": (temperature if temperature > 1e-4 else 0),
            "messages": messages,
        },
    )
    #print(res)
    output = res.json()["choices"][0]["message"]["content"]

    return output.strip() if output is not None else None


# def loadHugeLLM(session_len=1048576, tp=4):
#     backend_config = TurbomindEngineConfig(
#         rope_scaling_factor=2.5,
#         session_len=104857, #6,  # 1M context length
#         max_batch_size=1,
#         cache_max_entry_count=0.7,
#         tp=1) #XY4)  # 4xA100-80G.
#     pipe = pipeline('internlm/internlm2_5-7b-chat-1m', backend_config=backend_config)
#     return pipe


def inject_references_to_messages(messages, references):
    messages = copy.deepcopy(messages)

    system = f"""最新のユーザー クエリに対するさまざまなオープンソース モデルからの一連の応答が提供されています。あなたの仕事は、これらの応答を 1 つの高品質な応答に合成することです。これらの回答で提供された情報には偏りや不正確なものがある可能性があることを認識し、批判的に評価することが重要です。回答は、与えられた回答を単に再現するだけでなく、指示に対する洗練された正確かつ包括的な回答を提供する必要があります。応答が適切に構成され、一貫性があり、最高の精度と信頼性の基準に準拠していることを確認してください。

モデルからの応答:"""

    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    client,
    references,
    max_tokens=2048,
    temperature=0.7,
    ):

    print("^^^^^^^")
    print('messages', messages)
    print('references', references)

    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    return client(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def process_fn(
    item,
    temperature=0.7,
    max_tokens=2048,
):
    model = item["model"]
    messages = item["instruction"]

    output = generate_with_references(
        model=model,
        messages=messages,
        client=generate_with_local_api,
        references=item["references"],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    print(f"\nFinished querying [bold]{model}.[/bold]")

    return {"output": output}


def convert_dict_to_list(data):
    result = []
    for i in range(len(data["model"])):
        item = {
            "instruction": data["instruction"][i],
            "references": data["references"][i],
            "model": data["model"][i]
        }
        result.append(item)
    return result


def main(temperature=0.7, rounds=3, max_tokens=2048):
    tmp_message = copy.deepcopy(welcome_message)
    for model in models:
        tmp_message += f"{tmp_message}\n- {model[0]}"
    print(Markdown(tmp_message))

    # model_reference = loadHugeLLM(session_len=104857, tp=1)
    instruction= "きのこの里とたけのこの山どっちが美味しい？"

    moa_app(instruction, temperature, rounds, max_tokens)


def moa_app(instruction, temperature=0.7, rounds=3, max_tokens=2048):
    num_proc = len(models)

    data = {
        "instruction": [[
            {"role": "system", "content": "あなたは優秀なアシスタントです"},
            {"role": "user", "content": instruction}]
            for _ in range(num_proc)],
        "references": [""] * num_proc,
        "model": [m for m in models],
        "temperature": [temperature] * num_proc,
        "max_tokens": [max_tokens] * num_proc,
    }

    ctx = mp.get_context('spawn')

    refs = [""] * num_proc
    for _ in range(rounds):
        with ctx.Pool(processes=num_proc) as pool:
            # 非同期にタスクを実行
            results = pool.map_async(process_fn, convert_dict_to_list(data))
            final_results = results.get()

        for i in range(num_proc):
            refs[i] = final_results[i]["output"]

        for i in range(num_proc):
            data["references"][i] = refs

    messages=[{"role":"system","content":"あなたはイカしたアシスタントです"},
                {"role":"user","content":instruction}]

    output = generate_with_references(
        model=models_reference,
        messages=messages,
        client=generate_with_local_api,
        references=refs,
        temperature=temperature,
        max_tokens=1024*4,
    )
    print('Final output')
    print(output)
    return output


if __name__ =="__main__":
    main()