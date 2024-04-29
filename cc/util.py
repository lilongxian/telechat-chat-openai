import gc
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Tuple
from .base import DeltaMessage, ChatCompletionResponse, ChatCompletionResponseStreamChoice


def apply_stopping_strings(reply, stop_strings) -> Tuple[str, bool]:
    stop_found = False
    for string in stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou: is completed, trim it
        for string in stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found


def predict_stream(model, tokenizer, model_id,
                   gen_params, generate_config=None, model_type=None):
    output = ""
    has_send_first_chunk = False
    for new_response in generate_stream(model, tokenizer, gen_params, generate_config, model_type):
        decoded_unicode = new_response["text"]
        print("decoded_unicode: ", decoded_unicode)
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode

        # When it is not a function call and the character length is> 7,
        # try to judge whether it is a function call according to the special function prefix
        if len(output) > 7:
            # Non-function call, direct stream output
            finish_reason = new_response["finish_reason"]

            # Send an empty string first to avoid truncation by subsequent next() operations.
            if not has_send_first_chunk:
                message = DeltaMessage(
                    content="",
                    role="assistant",
                    function_call=None,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=finish_reason
                )
                chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
                yield "{}".format(chunk.model_dump_json(exclude_unset=True))

            send_msg = delta_text if has_send_first_chunk else output
            has_send_first_chunk = True
            message = DeltaMessage(
                content=send_msg,
                role="assistant",
                function_call=None,
            )
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=finish_reason
            )
            chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    else:
        yield '[DONE]'


# telechat7b11b12b-chat
@torch.inference_mode()
def generate_stream(model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    params: dict,
                    generate_config=None, model_type=None):

    messages_ = params["messages"]
    messages = []
    for m in messages_:
        role, content, func_call = m.role, m.content, m.function_call
        messages.append({"role": role, "content": content})

    if "telechat" in model_type:
        for msg in messages:
            if msg["role"] == 'assistant':
                msg["role"] = "bot"
            value = msg['content']
            msg['input_ids'] = tokenizer(value)["input_ids"]
            msg['attention_mask'] = [1] * len(msg['input_ids'])
        gen = model.chat(tokenizer=tokenizer,
                         question=messages[-1]["content"],
                         history=messages[:-1],
                         generation_config=generate_config,
                         stream=True)
        response_temp = ""
        for answer, history_ in gen:
            response, stop_found = apply_stopping_strings(answer, ["<_observation>", "<_end>", "<_user>"])
            response_temp += response
            yield {
                "text": response_temp,
                "finish_reason": "function_call" if stop_found else None,
            }
            if stop_found:
                break

        ret = {
            "text": response_temp,
            "finish_reason": "stop",
        }
        yield ret

        gc.collect()
        torch.cuda.empty_cache()
    else:
        assert 0, "param error: 当前模型类型{}不支持！".format(model_type)

