# AgenTracer adaptation notice

This adapter follows the public input/output contract from:

- Paper: *AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?*
- Code: <https://github.com/bingreeky/AgenTracer>
- Local reference checkout: `/data/xhy/codes/work/AgenTracer`

The public repository currently releases a MetaGPT data-curation pipeline and
sample TracerTraj data, but not the AgenTracer-8B weights. LycheeMAS therefore
uses an injected model callback and does not vendor the MetaGPT repository.

The adapted fields are `mistake_step`, `mistake_agent`, `reason`, and
`suggested_fix`. No source file is copied verbatim.

## LycheeMAS runner

The integration runner accepts the released Who&When parquet files:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_agentracer_attribution.py \
  --input /path/to/Algorithm-Generated.parquet \
  --output runs/agentracer/algorithm.jsonl \
  --model-path /path/to/model
```

`--smoke` uses gold labels only to validate the adapter and must not be used as
a model-quality result. The runner writes `agent_exact`, `step_exact`,
`joint_exact`, and mean `step_distance` to the companion metrics JSON.
