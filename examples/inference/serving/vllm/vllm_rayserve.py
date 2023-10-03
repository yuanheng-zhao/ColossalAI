"""Example of placing vllm engine as serve deployment"""
import json
import logging
import os
from typing import Any, AsyncGenerator

from pydantic import BaseModel
from ray import serve
from ray.serve import Application
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid
except:
    raise ImportError("Cannot import vLLM")


ray_serve_logger = logging.getLogger("ray.serve")


class GenConfigArgs(BaseModel):
    """Config for generation"""

    path: str
    tp_size: int = 2
    # max_batch_size: int = 4
    # max_input_len: int = 128
    # max_output_len: int = 32
    # engine_use_ray: bool = False # use Ray to start the LLM engine in a separate process as the server process.
    # max_num_batched_tokens: int = 2560 # maximum number of batched tokens per iteration
    # max_num_seqs: int = 8 # maximum number of sequences per iteration


@serve.deployment(ray_actor_options={"num_gpus": 0})
class EngineDriver:
    def __init__(self, config: GenConfigArgs) -> None:
        model_path = config.path
        tp_size = config.tp_size

        args = AsyncEngineArgs(model=model_path, gpu_memory_utilization=0.2, tensor_parallel_size=tp_size)
        self.engine = AsyncLLMEngine.from_engine_args(args)

    # Adapted from vllm.entrypoints.api_server.py
    # https://github.com/vllm-project/vllm/blob/2d1e86f1b15396119321cfb3a77acde72b0c08ee/vllm/entrypoints/api_server.py
    async def stream_results(self, results_generator) -> AsyncGenerator[bytes, None]:
        """For streaming case"""
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    async def add_request(self, request_id: str, prompt: str, sampling_params: SamplingParams) -> None:
        await self.engine.add_request(request_id, prompt, sampling_params)

    async def abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    async def __call__(self, request: Request) -> Any:
        # Refer to vllm.examples.api_client.py
        # prompt
        # n
        # use_beam_search
        # temperature
        # max_tokens
        # stream
        request_dict = await request.json()
        prompt = request_dict.pop("prompt")
        stream = request_dict.pop("stream", False)
        sampling_params = SamplingParams(**request_dict)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        # 1. Streaming Case
        if stream:
            task = BackgroundTask(self.abort_request, request_id)
            return StreamingResponse(self.stream_results(results_generator), background=task)

        # 2. Non-streaming case
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return Response(status_code=499)
            final_output = request_output

        assert final_output is not None
        prompt = final_output.prompt
        text_outputs = [prompt + output.text for output in final_output.outputs]
        ret = {"text": text_outputs}
        return JSONResponse(ret)


def app(args: GenConfigArgs) -> Application:
    print(args)
    if args.path is None or not os.path.exists(args.path):
        raise ValueError("Model path not provided or invalid path!")

    return EngineDriver.options(name="vLLM-Inference-Driver").bind(config=args)


# RAY_DEDUP_LOGS=0 serve run Colossal_Inference_rayserve:app path="PATH_TO_YOUR_MODEL_DIR"
