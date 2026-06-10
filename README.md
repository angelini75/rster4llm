# rster4llm

Research notes on optimal ingestion of geospatial data — rasters, vectors and imagery — into Large Language Model (LLM) environments, with a focus on Digital Soil Mapping (DSM) and Earth Observation workflows.

![Graphical abstract — geospatial data compressed into LLM-ready representations, with function calling over cloud-native ground-truth storage](graphical-abstract.svg)

The repository collects three independent deep-research syntheses on the same question, produced with different AI research assistants, for comparison and cross-validation:

| Document | Source |
|----------|--------|
| [Optimal Spatial Data Ingestion for LLMs — A Technical Synthesis](Resources/Optimal%20Spatial%20DataIngestion%20for%20LLMs%20--A%20Technical%20Synthesis%20by%20Claude.md) | Claude |
| [Architecting the Spatial-Semantic Bridge](Resources/Optimal%20Spatial%20Data%20Ingestion%20for%20LLMs%20by%20Gemini.md) | Gemini |
| [Optimal Data Formats for Spatial Data Ingestion into LLMs](Resources/Optimal%20Data%20Formats%20for%20Spatial%20Data%20Ingestion%20into%20LLMs%20by%20ChatGPT.md) | ChatGPT |

## Converging findings

The three syntheses independently converge on the same architecture:

1. **Never serialize raster pixels as text.** Compress rasters into ~100–150 token statistical summaries (per-band stats, histograms, derived indices) or into embeddings from geospatial foundation models (Clay, Prithvi-EO, SatCLIP).
2. **Token-efficient serialization for vectors.** WKT for geometries (~2.3× fewer tokens than GeoJSON); compact tabular formats (TOON/CSV) for attribute tables.
3. **Spatial-RAG retrieval.** Hybrid queries combining spatial filtering (PostGIS, H3 indexing) with semantic ranking (pgvector), fused via Reciprocal Rank Fusion.
4. **Cloud-native storage as ground truth.** COG for imagery, GeoParquet for vectors, Zarr for data cubes, catalogued with STAC.
5. **LLMs access data through function-calling tools** (catalog search, zonal statistics, point queries) — never raw pixels — with every answer traceable to its source. Decision support, not automated decision-making.
