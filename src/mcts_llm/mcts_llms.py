from .mctsr import MCTSr, MCTSNode
from .llm import openai_chat_completion
from .prompt_configs import (
    llama_3_8b_prompt_config,
    gpt_4o_prompt_config,
    RefineResponse,
)

class MCTSrLlama38B(MCTSr):
    def zero_shot(self) -> str:
        response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "The user will provide a problem. Solve the problem. Think step by step.",
                },
                {
                    "role": "user",
                    "content": f"<problem>\n{self.problem}\n</problem>",
                },
            ],
            model=llama_3_8b_prompt_config.model,
            base_url=llama_3_8b_prompt_config.base_url,
            max_tokens=4000,
        )
        assert response.choices[0].message.content is not None
        return response.choices[0].message.content

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        critique_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": llama_3_8b_prompt_config.critic_system_prompt, # 对当前答案进行批评与反思
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                        ]
                    ),
                },
            ],
            model=llama_3_8b_prompt_config.model,
            base_url=llama_3_8b_prompt_config.base_url,
            max_tokens=4000,
        )
        critique = critique_response.choices[0].message.content
        assert critique is not None
        self.critiques.append(critique)

        refined_answer_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": llama_3_8b_prompt_config.refine_system_prompt, # 基于反思的结果，对当前的答案进一步优化
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                            f"<critique>\n{critique}\n</critique>",
                        ]
                    ),
                },
            ],
            model=llama_3_8b_prompt_config.model,
            base_url=llama_3_8b_prompt_config.base_url,
            max_tokens=4000,
        )
        refined_answer = refined_answer_response.choices[0].message.content
        assert refined_answer is not None
        self.refinements.append(refined_answer)

        return MCTSNode(answer=refined_answer, parent=node)

    def _evaluate_answer(self, node: MCTSNode) -> int:
        messages = [
            {
                "role": "system",
                "content": llama_3_8b_prompt_config.evaluate_system_prompt, # 对当前答案进行评估
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<answer>\n{node.answer}\n</answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3): # 三次尝试，有一次成功就返回
            try:
                response = openai_chat_completion(
                    messages=messages,
                    model=llama_3_8b_prompt_config.model,
                    base_url=llama_3_8b_prompt_config.base_url,
                    max_tokens=4000,
                )
                assert response.choices[0].message.content is not None
                return int(response.choices[0].message.content)
            except ValueError:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        },
                        {
                            "role": "user",
                            "content": "Failed to parse reward as an integer.",
                        },
                    ]
                )
                if attempt == 2:
                    raise


class MCTSrGPT4o(MCTSr):
    def zero_shot(self) -> str:
        response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "The user will provide a problem. Solve the problem. Think step by step.",
                },
                {
                    "role": "user",
                    "content": f"<problem>\n{self.problem}\n</problem>",
                },
            ],
            model=gpt_4o_prompt_config.model,
            max_tokens=4000,
        )
        assert response.choices[0].message.content is not None
        return response.choices[0].message.content

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        critique_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": gpt_4o_prompt_config.critic_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                        ]
                    ),
                },
            ],
            model=gpt_4o_prompt_config.model,
            max_tokens=4000,
        )
        critique = critique_response.choices[0].message.content
        assert critique is not None
        self.critiques.append(critique)

        refined_answer_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": gpt_4o_prompt_config.refine_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                            f"<critique>\n{critique}\n</critique>",
                        ]
                    ),
                },
            ],
            model=gpt_4o_prompt_config.model,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        refined_answer = RefineResponse.model_validate_json(
            refined_answer_response.choices[0].message.content
        )
        self.refinements.append(refined_answer)

        return MCTSNode(
            answer=f"# Thought {refined_answer.thought}\n\n# Answer\n{refined_answer.answer}",
            parent=node,
        )

    def _evaluate_answer(self, node: MCTSNode) -> int:
        messages = [
            {
                "role": "system",
                "content": gpt_4o_prompt_config.evaluate_system_prompt,
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<answer>\n{node.answer}\n</answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3):
            try:
                response = openai_chat_completion(
                    messages=messages,
                    model=gpt_4o_prompt_config.model,
                    max_tokens=4000,
                )
                assert response.choices[0].message.content is not None
                return int(response.choices[0].message.content)
            except ValueError:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        },
                        {
                            "role": "user",
                            "content": "Failed to parse reward as an integer.",
                        },
                    ]
                )
                if attempt == 2:
                    raise


