"""Environments and the multi-turn rollout loop for nanoRL.

An Env is text-in, text-out and stateful:
    initial_user_msg = env.reset(sample)        # sample is a {question, answer, ...} dict
    obs, reward, done = env.step(action_text)   # action is a model response

Built-in envs:
  - SingleTurnEnv(system, reward_fn): wraps a (system_prompt, reward_fn) task into a 1-turn env
  - ToolEnv: parses <tool>args</tool> XML tags and dispatches to subclass `_tool_<name>`,
             terminates on <answer>X</answer>; subclasses provide `_score(answer_text)`
  - CalculatorEnv(ToolEnv): one tool, <calc>expr</calc>, evaluated in a sandboxed `eval`
  - PythonEnv(ToolEnv): one tool, <python>code</python>, run in a tiny `exec` sandbox

`rollout_one(env, tok, infer_url, sample, ...)` runs the multi-turn loop and returns a
dict with the full token sequence, a per-token loss_mask (1 on tokens the model generated,
0 on prompt + observations), the rollout-time logprobs (only meaningful at mask=1 positions),
and the final scalar reward.
"""
import contextlib
import io
import re
from abc import ABC, abstractmethod

import requests

# --- Env interfaces ----------------------------------------------------------

class Env(ABC):
    """Stateful single-prompt episode."""
    SYSTEM = ""
    MAX_TURNS = 1

    @abstractmethod
    def reset(self, sample: dict) -> str: ...        # initial user message

    @abstractmethod
    def step(self, action: str) -> tuple[str, float, bool]: ...  # (obs, reward, done)


class SingleTurnEnv(Env):
    """Wrap a (system, reward_fn) pair into a 1-turn Env. Lets the multi-turn rollout
    loop transparently handle 'classic' single-response tasks."""
    MAX_TURNS = 1

    def __init__(self, system: str, reward_fn):
        self.SYSTEM = system
        self.reward_fn = reward_fn
        self._answer = None

    def reset(self, sample):
        self._answer = sample["answer"]
        return sample["question"]

    def step(self, action):
        return "", self.reward_fn(action, self._answer), True


# Match <tag>...</tag> for any tag name; first capture is the tag name.
TOOL_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class ToolEnv(Env):
    """Multi-turn env with a tool registry: subclasses define `_tool_<name>(args)` for
    each tool and `_score(answer_text)` for terminal scoring. The model emits actions like
        <calc>2+2</calc>
        ... reasoning ...
        <answer>4</answer>
    """
    MAX_TURNS = 5

    def __init__(self):
        self._answer = None
        self._turn = 0

    def reset(self, sample):
        self._answer = sample["answer"]
        self._turn = 0
        return sample["question"]

    def step(self, action):
        self._turn += 1
        m = ANSWER_RE.search(action)
        if m:
            return "", self._score(m.group(1).strip()), True
        m = TOOL_RE.search(action)
        if m:
            name, args = m.group(1), m.group(2)
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                obs = f"<tool_error>unknown tool: {name}</tool_error>"
            else:
                try:
                    obs = f"<tool_result>{handler(args)}</tool_result>"
                except Exception as e:
                    obs = f"<tool_error>{type(e).__name__}: {e}</tool_error>"
        else:
            obs = ("<tool_error>no tool call or answer found; emit a tool tag like "
                   "<calc>...</calc> or finalize with <answer>...</answer></tool_error>")
        return obs, 0.0, self._turn >= self.MAX_TURNS

    @abstractmethod
    def _score(self, answer_text: str) -> float: ...


def _numeric_match(pred, target):
    pred = pred.replace(",", "").replace("$", "").strip()
    try:
        return float(abs(float(pred) - float(target)) < 1e-4)
    except ValueError:
        return float(pred == target)


class CalculatorEnv(ToolEnv):
    """Math problems with one tool: <calc>expr</calc>. Finalize with <answer>X</answer>."""
    SYSTEM = (
        "You are a careful math assistant. You may invoke a calculator by writing\n"
        "<calc>expression</calc>. The result will be returned to you as <tool_result>...</tool_result>.\n"
        "When you have the final numeric answer, write <answer>NUMBER</answer>."
    )

    def _tool_calc(self, expr):
        return str(eval(expr.strip(), {"__builtins__": {}}, {}))

    def _score(self, ans):
        return _numeric_match(ans, self._answer)


class PythonEnv(ToolEnv):
    """Math problems with a tiny python sandbox: <python>code</python>. Use print() for output."""
    SYSTEM = (
        "You are a careful math assistant. You may run Python by writing\n"
        "<python>code</python>. Use print() to see output, returned as <tool_result>...</tool_result>.\n"
        "When you have the final numeric answer, write <answer>NUMBER</answer>."
    )
    _BUILTINS = {"print": print, "len": len, "range": range, "abs": abs,
                 "min": min, "max": max, "sum": sum, "round": round,
                 "int": int, "float": float, "pow": pow, "divmod": divmod}

    def _tool_python(self, code):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": self._BUILTINS}, {})
        return buf.getvalue().strip() or "(no output)"

    def _score(self, ans):
        return _numeric_match(ans, self._answer)

# --- Multi-turn rollout (used by both train.py and eval.py) -------------------

def rollout_one(env, tok, infer_url, sample, *, max_response_tokens, temperature, top_p,
                max_total_len):
    """Run a multi-turn trajectory. Single-turn envs degenerate to one /generate call.

    Returns dict(all_ids, loss_mask, logprobs, reward) where:
      - all_ids:    full token sequence (prompt + every turn's gen + every env response)
      - loss_mask:  1 on tokens generated by the model, 0 elsewhere (prompt, env obs)
      - logprobs:   rollout-time logprob of each generated token; 0 outside mask=1 positions
      - reward:     scalar from env's terminal step()
    """
    init_user = env.reset(sample)
    messages = ([{"role": "system", "content": env.SYSTEM}] if env.SYSTEM else []) + \
               [{"role": "user", "content": init_user}]
    all_ids = list(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
    loss_mask = [0] * len(all_ids)
    logprobs = [0.0] * len(all_ids)
    reward = 0.0

    for _ in range(env.MAX_TURNS):
        budget = min(max_response_tokens, max_total_len - len(all_ids))
        if budget <= 0:
            break
        r = requests.post(f"{infer_url}/generate", json=dict(
            prompts=[all_ids], n=1, max_tokens=budget,
            temperature=temperature, top_p=top_p,
            stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
        ), timeout=600).json()
        gen_ids = r["response_ids"][0][0]
        gen_lp = r["response_logprobs"][0][0]
        all_ids = all_ids + gen_ids
        loss_mask = loss_mask + [1] * len(gen_ids)
        logprobs = logprobs + gen_lp

        action = tok.decode(gen_ids)
        obs, reward, done = env.step(action)
        if done:
            break

        # Append the env response by re-templating and taking the suffix delta. Round-trip
        # tokenization is stable for modern BPE/BBPE tokenizers (Qwen, Llama). Any mismatch
        # tokens land at mask=0 positions and don't affect the loss.
        messages.append({"role": "assistant", "content": action})
        messages.append({"role": "user", "content": obs})
        new_full = list(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
        added = new_full[len(all_ids):]
        all_ids = all_ids + added
        loss_mask = loss_mask + [0] * len(added)
        logprobs = logprobs + [0.0] * len(added)

    return dict(all_ids=all_ids, loss_mask=loss_mask, logprobs=logprobs, reward=reward)
