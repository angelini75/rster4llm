# Optimal spatial data ingestion for LLMs: A technical synthesis

The most effective strategy for ingesting complex spatial data into LLM environments combines **pre-computed embeddings from geospatial foundation models** (Prithvi, Clay) with **statistical summarization** for rasters, while using **GeoJSON or WKT serialization** for vector data. For text-only LLMs, statistical summaries reduce a full Sentinel-2 scene from millions of tokens to approximately **100-150 tokens** while preserving semantic content. Multimodal LLMs can process satellite imagery directly, but at significant token cost—GPT-4V consumes up to **1,700 tokens per image** in high-resolution mode. The emerging **Spatial-RAG** framework (Yu et al., 2025) now provides a rigorous methodology combining sparse spatial filtering with dense semantic matching, representing the current state-of-the-art for geospatial question answering.

The geospatial AI field has undergone rapid transformation since 2023 with foundation models like **Clay** (632M parameters) and **Prithvi-EO-2.0** (600M parameters) establishing standardized tokenization approaches using **16×16 or 8×8 pixel patches** on 256×256 chips. These models output 768-dimensional embeddings that compress raw imagery by **10-100× while preserving semantic information**—making them ideal intermediaries between raw spatial data and LLM context windows.

---

## Text-only LLMs demand serialized formats with careful token budgeting

When working with text-only LLMs, spatial data must be serialized into text representations, with significant implications for token efficiency and information preservation. **Well-Known Text (WKT)** emerges as the most token-efficient format for geometry serialization, requiring approximately **2.3× fewer tokens** than equivalent GeoJSON representations. A simple point coordinate in WKT (`POINT(125.6 10.1)`) consumes roughly 8 tokens versus 18 tokens for GeoJSON's verbose JSON structure.

For raster data, direct pixel serialization is computationally intractable—a single 256×256 RGB image would require approximately **20,000-60,000 tokens** if serialized as text. Instead, statistical summarization provides a practical alternative. Pre-computing per-band statistics (mean, standard deviation, percentiles), histogram distributions with 10-20 bins, and derived indices like NDVI compresses an entire satellite scene to a **100-150 token Markdown table** while preserving the information needed for most analytical tasks. Studies show GPT-4 achieves **0.66 accuracy on topological relation prediction** from WKT representations using few-shot prompting, demonstrating that LLMs can reason spatially from serialized formats.

The choice between formats depends critically on the analytical task. GeoJSON excels at preserving rich attribute metadata through its nested properties structure and is universally supported by LLMs extensively trained on JSON. However, GeoJSON lacks explicit topology encoding—proximity and adjacency relationships must be inferred computationally rather than read directly. **TopoJSON** extends GeoJSON with explicit topological encoding through shared arcs, reducing redundancy where features share boundaries (California-Nevada shared boundaries encoded once rather than duplicated). For digital soil mapping applications, covariates should be stored as statistical summaries organized by SCORPAN dimensions (soil, climate, organisms, relief, parent material, age, spatial position), enabling LLMs to reason about soil-landscape relationships without processing raw raster pixels.

---

## Multimodal LLMs enable direct imagery processing with controllable token costs

Vision-language models fundamentally change the spatial data ingestion paradigm by accepting imagery directly, though token economics remain crucial. OpenAI's GPT-4V operates in two modes: **low-resolution mode** consumes a fixed 85 tokens regardless of image complexity, while **high-resolution mode** tiles images into 512×512 segments at **170 tokens per tile plus 85 base tokens**. A 1024×1024 satellite image thus requires approximately 765 tokens (4 tiles × 170 + 85). Claude's approach scales linearly as **(width × height) / 750**, making a 1092×1092 image cost roughly 1,600 tokens.

Current multimodal LLMs demonstrate strong performance on general scene interpretation but struggle with specialized geospatial tasks. GPT-4V shows "exceptional proficiency" in flood severity assessment and basic land cover classification according to 2024 hydrological studies, but all tested models underperform on precise coordinate extraction, quantitative spatial measurements, and multi-spectral band interpretation. Gemini Pro Vision has exhibited concerning hallucination behaviors when analyzing maps, including fabricated geographic features in independent testing.

The geospatial AI community has developed specialized vision encoders that significantly outperform generic models on remote sensing tasks. **Clay Foundation Model v1.5** employs a dynamic embedding block that processes imagery from any number of spectral bands by encoding central wavelength values—enabling a single model to handle Sentinel-2 (10 bands), Landsat (6 bands), NAIP (4 bands), and SAR data interchangeably. The model uses **8×8 pixel patches** on 256×256 chips, incorporating wavelength-conditioned embeddings, GSD-scaled positional encoding, and latitude/longitude geographic encoding. **Prithvi-EO** from IBM/NASA takes a temporal approach, treating multi-temporal satellite data as video (B, C, T, H, W format) with 3D tubelet embeddings that encode spatial-temporal content jointly.

For integration with general-purpose LLMs, **GeoChat** (CVPR 2024) provides a template: a CLIP-ViT-L/14 visual backbone at 336×336 resolution connected to LLaVA-1.5 via a 2-layer MLP, fine-tuned on **318K remote sensing instruction-following samples** using LoRA. This architecture enables grounded conversations about satellite imagery, region-level visual question answering, and scene classification.

---

## Retrieval-augmented generation requires spatial-aware chunking and hybrid embeddings

Spatial-RAG represents the most sophisticated approach to geospatial question answering, with the **Spatial-RAG framework** (Yu et al., February 2025) establishing multi-objective optimization that balances spatial and semantic relevance through Pareto-optimal candidate selection. This architecture combines sparse spatial filtering (using geographic indices to narrow the search space) with dense semantic matching (vector similarity on embeddings).

For chunking strategies, **H3 hexagonal indexing** (Uber) emerges as the optimal choice for spatial partitioning. H3's equal-distance neighbor property (6 equidistant neighbors per cell) eliminates directional bias present in rectangular grids, while its **16 resolution levels** spanning ~85M km² to ~0.7 cm² cells accommodate scales from global analysis to building-level precision. For urban POI applications, resolution 9-11 (~100m-500m cells) provides appropriate granularity. Google's **S2 geometry** offers an alternative using Hilbert curves that provide better spatial locality than geohash's Z-order curve, with 30 resolution levels and an efficient region covering algorithm.

Embedding strategies should combine multiple modalities for optimal retrieval. **SatCLIP** (Microsoft Research, AAAI 2024) provides state-of-the-art location-aware embeddings using spherical harmonics positional encoding fed through SIREN neural networks, trained contrastively to match Sentinel-2 imagery with GPS coordinates. The recommended hybrid embedding approach concatenates:

- **Text embeddings** of spatial metadata (using sentence-transformers or OpenAI embeddings)
- **Location embeddings** from SatCLIP or spherical harmonics encoding
- **Image embeddings** from geospatial foundation models for visual context

Vector database selection significantly impacts system capabilities. **PostgreSQL with PostGIS and pgvector** provides the most comprehensive solution for SQL-based workflows, combining full spatial operators (ST_Distance, ST_Within, ST_Buffer) with HNSW and IVFFlat vector indexes in a single query engine. A hybrid query can filter spatially (`ST_DWithin(location, query_point, 5000)` for 5km radius) before performing vector similarity search, then rank results using **Reciprocal Rank Fusion** (RRF): `final_score = 1/(spatial_rank + k) + 1/(semantic_rank + k)`. For high-performance requirements, **Qdrant** offers native geo-filtering (geo_distance, geo_bounding_box) with strong hybrid search capabilities, achieving approximately **41 QPS at 99% recall** on 50M vectors with filtering.

---

## GeoParquet and cloud-optimized formats bridge storage and LLM pipelines

The format landscape has consolidated around cloud-native standards that enable efficient partial reads and rich metadata preservation. **Cloud-Optimized GeoTIFFs (COGs)**, ratified as an OGC Standard in July 2023, leverage internal tiling (256×256 or 512×512 pixels) and reduced-resolution overviews to enable HTTP range requests for targeted data access. COGs require **2× less data transfer and ~25× fewer GET requests** compared to JPEG2000 for equivalent partial reads according to Element 84 benchmarks. GDAL's virtual filesystems (`/vsicurl/`, `/vsis3/`, `/vsigs/`, `/vsiaz/`) enable streaming access to COGs stored on any major cloud provider.

**GeoParquet v1.1.0** (released June 2024, OGC incubating) provides columnar storage with comprehensive geospatial metadata including full `projjson` CRS definitions, WKB geometry encoding, and row-group level bounding box statistics enabling spatial filtering during queries. Column pruning reduces data transfer to only required attributes, while compression achieves significant storage reduction for coordinate-heavy data. The format has achieved rapid adoption across **20+ tools in 6 languages** including GeoPandas, DuckDB, Apache Sedona, BigQuery, and Snowflake. **GeoParquet 2.0** planning aligns with native Parquet GEOMETRY/GEOGRAPHY types added to the Parquet specification in March 2025.

For multidimensional data, **Zarr** provides cloud-native chunked array storage ideal for time-series and multispectral data cubes. Unlike NetCDF, Zarr stores metadata separately as lightweight JSON and chunks as independent objects, enabling thread-safe parallel retrieval optimized for HTTP range requests. **VirtualiZarr** (announced AGU 2024) enables accessing legacy NetCDF as virtual Zarr stores without data conversion—demonstrated on 40TB/500,000 NetCDF files. AWS benchmarks show Zarr queries completing in minutes where equivalent NetCDF queries required hours for ERA5 time-series analysis.

**STAC (SpatioTemporal Asset Catalog)** provides the metadata layer connecting these formats to LLM pipelines. STAC's JSON structure—Items (GeoJSON features with metadata), Collections (grouped items), and Catalogs (hierarchical organization)—is directly parseable by LLMs and ideal for function-calling interfaces. An LLM can invoke STAC API searches to discover relevant datasets, then retrieve specific COG tiles for detailed analysis. NASA's CMR-STAC, Microsoft Planetary Computer, and AWS Earth Search all expose STAC APIs covering hundreds of petabytes of geospatial data.

---

## Geospatial foundation models establish emerging tokenization standards

The geospatial foundation model landscape has rapidly matured, establishing de facto standards for spatial tokenization. The **16×16 or 8×8 pixel patch** applied to **256×256 pixel chips** has emerged as the dominant tokenization approach, with chips at 10m resolution (Sentinel-2) covering approximately **2.56 km² per embedding**. This produces **768-dimensional embeddings** per chip across most major models.

**Prithvi-EO-2.0** (600M parameters) from IBM/NASA processes 6 HLS bands (Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2) using 16×16 patches with temporal attention across time steps, pre-trained on **4.2 million globally distributed samples** using masked autoencoder (MAE) learning with 75% masking ratio. The model achieved **75.6% average score on GEO-Bench**, representing an 8% improvement over v1.0.

**Clay v1.5** (632M parameters) innovates with wavelength-conditioned dynamic embeddings that accept any combination of spectral bands by encoding central wavelength values. Pre-trained on **70 million globally distributed image chips** with LULC-stratified sampling, Clay combines 95% MAE reconstruction loss with 5% DINOv2 representation distillation. The model supports Sentinel-2, Landsat 8/9, Sentinel-1 SAR, NAIP, MODIS, and high-resolution imagery from a single architecture.

**DOFA** (Dynamic One-For-All) pushes modality-agnosticism further with hypernetworks that dynamically generate encoder weights based on input wavelengths, enabling neural-plasticity-inspired adaptation across optical, SAR, and hyperspectral sensors. **SpectralGPT** addresses spectral continuity with 3D masked transformers operating on volumetric (spatial×spatial×spectral) tokens at **90% masking ratio**, preserving band-to-band relationships that patch-based approaches can fragment.

Location encoding has converged on **spherical harmonics** for global applications (SatCLIP, with scale parameter L controlling granularity) or **random Fourier features** for more localized systems (GeoCLIP). These encodings map continuous coordinates to learned representations that capture geographic semantics—the embedding for a tropical rainforest location encodes expected vegetation patterns before any imagery is processed.

---

## Recommended ingestion pipeline balances current compatibility with future-proofing

Based on this synthesis, a standardized ingestion pipeline should follow a tiered architecture:

**Tier 1 - Raw Storage**: Store original data in cloud-optimized formats (COGs for imagery, GeoParquet for vectors, Zarr for multidimensional data) with comprehensive STAC metadata catalogs. This preserves full fidelity for future reprocessing as foundation models improve.

**Tier 2 - Embedding Generation**: Process imagery through geospatial foundation models (Clay or Prithvi) to generate 768-dimensional embeddings per 256×256 chip. Store embeddings in a vector database (pgvector+PostGIS for SQL workflows, Qdrant for performance-critical applications) with H3 cell IDs at resolution 9-10 as metadata for spatial filtering.

**Tier 3 - Semantic Summarization**: Generate statistical summaries (per-band statistics, derived indices, zonal aggregations) and natural language descriptions for context-limited LLM interactions. Store these as STAC Item properties or companion JSON documents.

**Tier 4 - RAG Integration**: Implement Spatial-RAG architecture combining H3-based spatial pre-filtering with hybrid embeddings (text metadata + SatCLIP location + GeoFM imagery). Use RRF to merge spatial and semantic ranking for final retrieval.

For **text-only LLMs**, serialize vectors as WKT (2.3× more efficient than GeoJSON), simplify geometries before serialization, and provide statistical summaries rather than raw raster values. For **multimodal LLMs**, resize imagery to 1024×1024 maximum, include bounding box and CRS in prompts, and use low-resolution mode for scene understanding tasks where precise details are not critical.

The architecture should expose **STAC search, COG tile extraction, and zonal statistics as LLM function tools**, enabling autonomous data discovery and retrieval. As geospatial foundation models continue advancing toward unified architectures handling arbitrary sensors and scales, the embedding layer provides a stable interface that can be updated independently of the LLM integration layer—ensuring investments in pipeline infrastructure remain valuable as both LLM and GeoFM capabilities evolve.