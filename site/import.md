# Import and use AGRIBALYSE

The existing Sentier adapter is the starting point. The project spans releases; the current documented adapter imports AGRIBALYSE 3.2 into Brightway 2.5 with an ecoinvent 3.9.1 background and EF 3.1 methods. Community participation does not require a private or unreleased adapter.

## Prepare the baseline

```sh
git clone https://github.com/sentier-dev/sentier-agribalyse.git
cd sentier-agribalyse
uv sync --extra test
```

The repository ships ADEME's export without its embedded ecoinvent datasets. Licensed users regenerate the background snapshots locally as described in [BOOTSTRAP.md](../BOOTSTRAP.md); the repository does not supply those. For scoring and Brightway export use the extras and platform-specific solver setup documented in [the repository README](../README.md).

The established tools build the mapping registry, link flows, register the 19 headline EF 3.1 methods, backtest against ADEME references and export a parity-checked Brightway package. Follow the version-specific README sequence; retain source hashes, method sources, mappings and parameter overrides as part of the baseline.

## Bring a candidate contribution

Use the existing MappingRegistry and parameter override tools. Work in an isolated checkout with separate source, cache, registry and output folders. Keep the baseline and candidate's background, units, allocation, boundary, geography and method fixed unless the intended change explicitly concerns one of them.

Read [application and testing](../community/application-and-testing.md), [build validation](../community/build-validation.md) and [the contribution workflow](../community/workflow.md). The characterization audit checks the final scoring biosphere before package emission; backtesting and export parity remain separate requirements.

## What can be public?

Publish code, permitted mappings, reproducible metadata and authorized aggregate comparisons. Licensed ecoinvent snapshots and packages remain local. Check [data and attribution](../community/data-and-attribution.md) before sharing any artifact.
