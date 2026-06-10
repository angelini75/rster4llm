# **Architecting the Spatial-Semantic Bridge: Optimization of Geospatial Data Ingestion Pipelines for Large Language Models**

## **Executive Summary**

The integration of geospatial intelligence with Large Language Models (LLMs) represents a paradigm shift in Earth Observation (EO) and Geographic Information Systems (GIS). However, this convergence is currently hindered by a fundamental "modality gap": LLMs operate on discrete sequences of text tokens, while geospatial data exists as continuous multidimensional signals (raster imagery) or complex geometric structures (vector data). This disconnect necessitates a rigorous re-evaluation of data ingestion architectures. We cannot simply feed raw coordinates or pixel arrays into a transformer’s context window without incurring catastrophic token costs and semantic degradation.

This report provides a comprehensive technical evaluation of the ingestion mechanisms required to bridge this gap, specifically addressing the requirements for Remote Sensing (RS) imagery, Digital Soil Mapping (DSM) covariates, and vector datasets. We analyze the efficacy of text serialization formats (WKT vs. GeoJSON vs. TOON), the emergence of visual-spatial foundation models (Prithvi, SatClip), and the architecture of hybrid Retrieval-Augmented Generation (RAG) systems. Based on a synthesis of current research, we propose a standardized "Spatial-Semantic Bridge" pipeline that leverages hierarchical spatial indexing (H3/S2), hybrid vector-SQL retrieval (PostGIS \+ pgvector), and projection-based multimodal embedding to maximize token efficiency, semantic preservation, and reasoning accuracy.

The optimal pipeline for 2025/2026 is a hybrid system: it serializes vector geometry via Well-Known Text (WKT) for topological precision, encodes attributes via Token-Oriented Object Notation (TOON) for efficiency, and projects raster data into the LLM’s latent space via soft-prompted geospatial foundation models like Prithvi-EO-2.0. This architecture moves beyond simple file conversion to establish a unified semantic plane where spatial reasoning can occur natively within the generative workflow.

## **1\. The Geospatial Modality Gap: Theoretical Underpinnings**

The central challenge in deploying LLMs for geospatial tasks lies in the translation of spatial concepts into the model's native embedding space. Standard Natural Language Processing (NLP) tokenizers (e.g., Byte-Pair Encoding or WordPiece) are ill-equipped to handle the high dimensionality of remote sensing data or the coordinate precision of vector geometries. Understanding this "modality gap" is a prerequisite for architecting an effective ingestion pipeline.

### **1.1 The Tokenization Bottleneck and Dimensionality Curse**

LLMs are sequence processors. They perceive the world as a linear stream of discrete tokens, optimized for the statistical structure of human language. Geospatial data, conversely, is inherently multi-dimensional and continuous. When we attempt to force spatial data into a text-based context window, we encounter the "Curse of Dimensionality" in a novel form: token exhaustion.

Vector data, representing discrete geometries (points, lines, polygons), relies on high-precision floating-point coordinates to maintain topological integrity. However, standard tokenizers fragment these numbers destructively. A coordinate pair like (-122.4194, 37.7749) does not map to a single semantic token. Instead, a tokenizer like OpenAI's cl100k\_base might fracture it into multiple sub-word units: (-, 122, ., 41, 94, ,, 37, ., 77, 49, ). This fragmentation destroys the numerical adjacency and mathematical significance of the coordinate.1 The model loses the inherent understanding that 37.7749 is spatially "close" to 37.7750 because the token sequences may be radically different despite the numerical proximity.

Furthermore, the scale of geospatial data dwarfs the capacity of even the largest context windows. A single Sentinel-2 satellite scene contains millions of pixels, each with multiple spectral bands. Serializing this raster data into text (e.g., a list of pixel values) is computationally prohibitive and semantically vacuous. Early experiments that attempted to describe images via dense captioning proved insufficient for expert-level tasks; a caption like "a forest next to a river" loses the high-frequency spatial information required for precision tasks like calculating Digital Soil Mapping (DSM) covariates or detecting subtle land-use changes.3

### **1.2 The Semantic Void: Geometry vs. Geography**

A second fundamental gap exists between **geometry** (the mathematical shape) and **geography** (the semantic meaning). A polygon defined by WKT POLYGON((0 0, 0 1, 1 1, 1 0, 0 0)) is geometrically complete but geographically meaningless without context. In traditional GIS, this context is provided by attribute tables and metadata. In an LLM environment, this context must be explicitly encoded into the prompt or the embedding.

Research indicates that the format of ingestion significantly impacts the reasoning capabilities of the model. While LLMs have demonstrated emergent spatial reasoning—such as deducing that a point is inside a polygon—their performance is highly sensitive to the verbosity and structure of the input format. The trade-off is often between **token efficiency** (minimizing context window usage to allow for more logic) and **structural explicitness** (preserving semantic relationships to prevent hallucination).5 For DSM specifically, where the relationship between a point observation (vector) and the underlying environmental covariates (raster slope, aspect, lithology) is causal, the ingestion pipeline must preserve these cross-modality links explicitly, or the LLM will fail to infer the soil properties correctly.

### **1.3 Multimodal Alignment**

The current state-of-the-art has shifted towards **Multimodal Large Language Models (MLLMs)** that utilize specialized visual encoders to bypass text serialization entirely for rasters. However, standard vision encoders (like CLIP) are trained on natural images (internet photos) and lack the capacity to interpret the specific spectral signatures and nadir perspectives of satellite imagery.7 A photo of a cornfield from a handheld camera looks vastly different from a multispectral nadir view from 700km altitude. Consequently, optimal ingestion requires domain-specific encoders capable of generating "spatial tokens" that represent geographic features rather than just visual textures.9

## **2\. Vector Data Ingestion: The Serialization Debate**

The question of how to represent vector data as text for LLM consumption is critical. Vector data is the "language" of GIS, describing discrete features like soil sample locations, field boundaries, or road networks. Two primary standards dominate the discourse: **Well-Known Text (WKT)** and **GeoJSON**. A third, emerging category includes optimized formats like **TOON** and **GeoTokens**.

### **2.1 Well-Known Text (WKT): The Efficiency Standard**

Well-Known Text (WKT) is an Open Geospatial Consortium (OGC) standard designed for compact ASCII representation of geometry. Its primary advantage in the context of LLMs is **token efficiency**. Because WKT strips away the hierarchical key-value structure inherent to JSON, it represents complex geometries with significantly fewer characters.

Benchmarks indicate that WKT can reduce token usage by **30-50%** compared to verbose GeoJSON for equivalent geometries.2 This is crucial for RAG applications where context window limits are a binding constraint. For example, when ingesting a dataset of DSM sample points to ask the LLM to identify spatial clusters, fitting 50% more data points into the context window directly correlates to a more robust statistical analysis by the model.

Beyond efficiency, WKT supports various **Coordinate Reference Systems (CRS)**. It can describe complex combinations of geocentric and projected systems, whereas GeoJSON is strictly limited to WGS 84 (latitude/longitude).11 This makes WKT the preferred format for high-precision engineering or surveying applications where reprojection to WGS 84 introduces unacceptable distortion or error. For instance, calculating the precise area of a land parcel or the distance between soil samples requires a projected metric CRS (like UTM), which WKT handles natively.

Studies utilizing the **GeoAnalystBench** framework suggest that while proprietary models (like GPT-4) can parse both formats, WKT's conciseness allows for the inclusion of more "few-shot" examples in the prompt, indirectly improving reasoning performance on complex topological tasks. The reduced syntactic noise allows the model's attention heads to focus on the coordinate values and the geometric logic rather than the repeated JSON keys.6

### **2.2 GeoJSON: The Semantic Standard**

GeoJSON, based on the JSON standard (RFC 7946), is the de facto format for web mapping and API data exchange. Its verbosity, while a drawback for token limits, offers superior **semantic clarity**.

The primary strength of GeoJSON is the **FeatureCollection** model, which treats properties (attributes) and geometry as a unified feature object. This structure aligns well with the training data of code-generation models. LLMs trained on GitHub repositories have seen billions of lines of JSON and Python dictionaries, making them highly proficient at parsing and manipulating this structure.13 If the LLM is acting as an agent generating Python code (e.g., using geopandas), GeoJSON is often the safer input format because it minimizes the need for intermediate parsing logic; the LLM can simply write gdf \= geopandas.read\_file('data.json') without needing to import shapely.wkt libraries or handle CRS strings manually.5

However, the repetitive structure of GeoJSON (repeated keys like "type": "Feature", "geometry":...) acts as "distractor noise" for the attention mechanisms of smaller LLMs. Research shows that models may hallucinate attributes or misinterpret nesting levels when processing deeply nested JSON structures, leading to errors in data extraction.12

### **2.3 Emerging Formats: TOON and GeoTokens**

To address the inefficiencies of JSON without losing its structural benefits, new formats like **Token-Oriented Object Notation (TOON)** have been proposed. TOON replaces the heavy syntax of braces, quotes, and repeated keys with indentation and header-based arrays.

Token Efficiency Analysis:  
Comparisons show TOON can reduce token counts by up to 60% compared to pretty-printed JSON and 23% compared to minified JSON.16

* *JSON:* \[{"id": 1, "soil\_type": "Loam"}, {"id": 2, "soil\_type": "Clay"}\]  
* *TOON:*  
  \# id, soil\_type  
  1, Loam  
  2, Clay

This format is particularly effective for ingesting tabular attribute data associated with vector geometries, such as the covariate tables in Digital Soil Mapping (e.g., slope, aspect, vegetation index values for each sample point). It allows the LLM to ingest dense numerical matrices without the overhead of CSV headers repeated or JSON syntax.15

**GeoTokens** represent a more radical shift: discretizing the geometry itself into **hierarchical spatial indices** like Uber's H3 or Google's S2. By representing a polygon as a set of H3 cell tokens, the data is converted into a sequence of discrete identifiers. This transforms the spatial reasoning problem into a sequence prediction problem, which aligns perfectly with the transformer architecture.18 H3 indices are fixed-length strings (e.g., 8928308280fffff) that carry implicit topological information; an LLM can learn that certain hex-strings are neighbors, enabling "fuzzy" spatial reasoning without floating-point arithmetic.20

### **2.4 Comparative Analysis of Vector Formats**

| Feature | Well-Known Text (WKT) | GeoJSON | TOON (Emerging) | H3/S2 Indices (GeoTokens) |
| :---- | :---- | :---- | :---- | :---- |
| **Token Efficiency** | High (30-50% savings vs JSON) | Low (Verbose syntax) | Very High (Header-based) | Variable (Resolution dependent) |
| **Human Readability** | Moderate | High | High | Low (Hex strings) |
| **Attribute Handling** | None (Geometry only) | Native (Feature Collection) | Native (Optimized for Tables) | Implicit (Requires external lookup) |
| **LLM Reasoning** | High for topology & precision | High for code generation | High for logic & data extraction | High for spatial proximity/clustering |
| **Standardization** | ISO Standard | RFC 7946 | Experimental | Industry Standard (Indices) |
| **DSM Suitability** | Excellent for sample locations | Good for final map delivery | Excellent for covariate tables | Excellent for gridded covariates |

## **3\. Raster Data Ingestion: Visual Encoders and Foundation Models**

Ingesting raster data (satellite imagery, DEMs, airborne hyperspectral) requires moving beyond text entirely. The "optimal" approach today leverages **Geospatial Foundation Models (GeoFMs)** that encode visual data into dense vector embeddings compatible with LLMs. This is particularly relevant for Digital Soil Mapping, which relies heavily on raster covariates like Digital Elevation Models (DEMs) and radiometric indices.

### **3.1 The Prithvi Architecture: A Temporal Vision Transformer**

The **Prithvi** model, developed by IBM and NASA, represents the state-of-the-art in ingest architecture for remote sensing. It is a **Temporal Vision Transformer (ViT)** pretrained on the Harmonized Landsat Sentinel-2 (HLS) dataset. Its architecture is specifically designed to handle the multi-dimensional nature of EO data.9

3D Patch Embeddings:  
Unlike standard ViTs (like those used in Stable Diffusion) that use 2D patches, Prithvi utilizes 3D patch embeddings to handle the temporal dimension. The input sequence of images is divided into non-overlapping cubes of size $(t, h, w)$. This allows the model to capture spatiotemporal dynamics—such as the seasonal wetting and drying cycles of soil, or the phenological stages of vegetation cover—natively.10 For DSM, this is critical; soil properties are often inferred from how the surface changes over time (e.g., bare soil composite selection).  
Positional and Metadata Encoding:  
Prithvi employs a sophisticated 3D positional encoding scheme. It generates independent 1D sin/cos encodings for time, height, and width, and fuses them. Crucially, it also encodes metadata (geolocation and acquisition time) and injects this as a bias into the embedding. This "location-aware" encoding ensures that the model understands that a pixel in a tropical rainforest is semantically different from a visually similar pixel in a boreal forest due to biome differences. This context is essential for global-scale soil mapping where climatic zones drive pedogenesis.10  
Masked Autoencoder (MAE) Mechanism:  
The pretraining objective involves masking a high percentage (e.g., 50%) of the input tokens and forcing the model to reconstruct them. This forces the model to learn robust internal representations of geospatial features, making the resulting embeddings highly information-dense. When used for ingestion, we extract these latent embeddings rather than the reconstructed image, providing the LLM with a rich, compressed representation of the surface.10

### **3.2 SatClip: Global Location Embeddings**

While Prithvi focuses on the *content* of the image (what is there?), **SatClip** focuses on the *location* (where is it?). SatClip is a contrastive learning model that aligns satellite imagery with geographic coordinates.24

Mechanism and Utility:  
SatClip learns a continuous representation of the globe where the embedding of a location $(lat, lon)$ is aligned with the visual features of that location observed from space. In an ingestion pipeline, SatClip embeddings serve as a powerful "positional anchor." By concatenating a SatClip embedding with the text query, the LLM can be grounded in a specific geographic context without needing explicit coordinate inputs or external climate data lookups. For DSM, a SatClip embedding acts as a proxy for the static environmental covariates (climate, topography, parent material) that define the "state factors" of soil formation.26

### **3.3 The Projection Layer: Bridging Modalities**

To feed these raster embeddings (from Prithvi or SatClip) into a text-based LLM, a **projection layer** is required. This is the bridge that translates "visual tokens" into "text tokens."

Soft Prompting:  
This technique involves freezing the LLM and training only the projection layer (usually a Multi-Layer Perceptron or Q-Former) to map the dimension of the visual embedding (e.g., 768 or 1024 dimensions) to the input dimension of the LLM (e.g., 4096 dimensions for Llama-3). This allows the LLM to "see" the satellite image as a sequence of semantic concepts. The projection layer learns to translate the high-dimensional feature vector of a "wetland" from Prithvi into the semantic token vector for "wetland" in the LLM's language space, enabling the model to reason about it textually.27

## **4\. Retrieval-Augmented Generation (RAG) for Geospatial**

Standard RAG pipelines retrieve text chunks based on semantic similarity. **Geo-RAG** (Geospatial RAG) must retrieve data based on both semantic similarity *and* spatial proximity. This dual requirement necessitates a specialized architecture.

### **4.1 Hybrid Retrieval Architectures**

The optimal Geo-RAG architecture utilizes a **Hybrid Search** mechanism that fuses sparse (keyword), dense (semantic), and spatial (geometric) retrieval.

PostgreSQL with PostGIS and pgvector:  
While specialized vector databases (like Pinecone or Weaviate) have gained popularity, PostgreSQL equipped with the PostGIS and pgvector extensions has emerged as the unified solution for this architecture. It is uniquely capable of executing complex logic that filters by precise geometry and ranks by semantic similarity in a single query plan.29  
Consider a query for DSM: "Find soil samples with high clay content in the floodplains of the Mississippi River." A hybrid query would:

1. **Spatial Filter (PostGIS):** Use ST\_Intersects to filter sample points that fall within the "Mississippi Floodplain" polygon.  
2. **Semantic Rank (pgvector):** Use cosine distance (\<=\>) to rank the remaining points based on the semantic similarity of their descriptions to "high clay content."  
3. **Result:** A list of spatially relevant and semantically matching records.

This consolidation reduces the complexity of maintaining separate vector and spatial databases (the "split-brain" problem) and ensures transactional integrity between spatial updates and vector index updates.31

### **4.2 Spatial Chunking Strategies**

The "chunking" of spatial data is more complex than splitting text documents. How do we divide continuous space for retrieval?

Hierarchical Indexing (H3/S2):  
Instead of arbitrary bounding boxes, data is indexed using H3 (hexagonal) or S2 (rectangular) hierarchical cells. Each cell becomes a "document" containing all features within it.

* **Grid-Based Chunking:** Features are bucketed into H3 cells (e.g., Resolution 7 or 8). This preserves spatial locality—features in the same cell are retrieved together.  
* **Hierarchical Summarization:** The most effective strategy involves a multi-level index. At a coarse level (H3 Res 4), cells contain summaries of the region (e.g., "Agricultural region with clay soils"). At a fine level (H3 Res 9), cells contain individual feature attributes. The LLM traverses this hierarchy based on the granularity of the user's query, retrieving summaries for broad questions and specific features for precise ones.33

## **5\. The "Spatial-Semantic Bridge" Recommendation**

Based on the synthesis of the above technologies, we define the optimal **Geospatial Ingestion Pipeline** for 2025/2026. This pipeline is designed to handle multimodal inputs—specifically incorporating DSM covariates—and provide grounded, hallucination-free outputs.

### **5.1 Architecture Diagram and Workflow**

The architecture consists of three parallel ingestion tracks (Vector, Raster, Text) that converge into a unified context for the LLM.

#### **Track A: Vector Ingestion (The Topological Track)**

* **Input:** Point data (soil samples), Polygons (field boundaries, soil units).  
* **Preprocessing:** Simplify geometries using the Visvalingam-Whyatt algorithm to reduce vertex count while preserving shape. Reproject to EPSG:4326 for storage, but retain original projection metadata.  
* **Serialization:**  
  * **Geometry:** Convert to **WKT** strings. This maximizes precision and token efficiency for the spatial component.  
  * **Attributes:** Serialize tabular attributes (e.g., pH, organic carbon, texture class) using **TOON**. This minimizes token overhead for the dense numerical data typical of DSM.  
* **Indexing:** Compute the **H3 Index** (Resolution 8-10) for each feature to enable fast spatial lookups.  
* **Storage:** Store in **PostGIS** with a GIST index for spatial querying and a pgvector column for semantic attribute search.

#### **Track B: Raster Ingestion (The Visual-Semantic Track)**

* **Input:** Satellite imagery (Sentinel-2), DEM (SRTM/Copernicus), Gridded Covariates (Gamma radiometrics).  
* **Tiling:** Tile large rasters into $256 \\times 256$ chips.  
* **Embedding Generation:**  
  * Pass multi-spectral imagery through **Prithvi-EO-2.0** to generate 3D spatiotemporal embeddings that capture phenology and spectral traits.  
  * Generate a **SatClip** embedding for the chip centroid to encode global location context.  
* **Projection:** Fuse Prithvi and SatClip embeddings and pass them through a trained **Linear Projection Layer** to align them with the LLM's text embedding space.  
* **Output:** A sequence of "Visual Tokens" representing the scene.

#### **Track C: Text/Metadata Ingestion (The Knowledge Track)**

* **Input:** Soil survey reports, metadata documentation, scientific papers.  
* **Chunking:** Standard semantic chunking, but enriched with **Spatial Tags** (H3 indices or bounding boxes) extracted from the text entity recognition.  
* **Storage:** Pgvector store.

#### **Track D: The Synthesis Layer (The LLM Agent)**

* **Prompt Assembly:** The LLM receives a prompt containing:  
  * System Instructions (Persona, Constraints).  
  * Retrieved Text Chunks (Track C).  
  * Serialized Vector Data (WKT \+ TOON from Track A).  
  * Visual Tokens (Projected Embeddings from Track B).  
* **Agentic Framework:** The LLM is wrapped in an agent (e.g., LangChain or LlamaIndex) capable of executing SQL calls to the PostGIS database. This allows the model to "ask" for more data ("Select soil samples within 1km of this embedding location") dynamically, rather than relying solely on the initial context window.35

### **5.2 Why This is Optimal**

This hybrid approach addresses the specific weaknesses of each modality:

1. **Efficiency:** It avoids the token cost of raster-to-text conversion by using **embeddings** for imagery.  
2. **Precision:** It avoids the precision loss of pure tokenization by maintaining a **PostGIS backend** for exact calculations (area, distance) which are then fed to the LLM as results.  
3. **Grounding:** It prevents hallucinations by grounding the LLM in **SatClip** location embeddings and **Prithvi** spectral embeddings.  
4. **Semantic Clarity:** It utilizes **TOON** for attributes, ensuring that the dense covariate tables required for DSM are ingested with maximum token efficiency.

## **6\. Emerging Frontiers: Large Spatial Models (LSMs)**

While the current recommendation relies on adapting general-purpose LLMs (like GPT-4 or Llama-3) to spatial tasks, the future lies in **Large Spatial Models (LSMs)** trained natively on geospatial data.7

### **6.1 Native Spatial Tokenization and World Simulators**

LSMs will move beyond adapting text tokenizers. They will be trained from scratch on "GeoTokens"—multimodal units that inherently contain spectral, semantic, and locational data. A GeoToken might represent a tuple: (Location\_H3, Time, Spectral\_Vector, Semantic\_Label).  
Unlike current models that process static data, LSMs will function as world simulators. They will be capable of predicting future states in an autoregressive manner. For example, given a sequence of satellite images and weather data for a region, the LSM could predict the next image frame, effectively simulating flood progression or crop growth. This capability will transform DSM from a static mapping exercise into a dynamic forecasting discipline, predicting soil moisture or erosion risk in near real-time based on the model's internal physics simulation.37

## **7\. Strategic Conclusions**

The integration of geospatial data into LLMs is moving from "text-based hacks" to "native multimodal integration." To build the optimal pipeline today:

1. **Abandon Pure Text for Rasters:** Cease trying to describe complex rasters with text captions. Use vision encoders (Prithvi) and projection layers to feed visual tokens directly to the LLM.  
2. **Embrace PostGIS/pgvector:** The vector database hype cycle often overlooks PostGIS. For geospatial RAG, the combination of pgvector and PostGIS is superior to specialized vector-only DBs due to the necessity of precise spatial filtering.  
3. **Standardize on WKT and TOON:** For text-based geometry exchange within the prompt, WKT is the efficiency champion. For attribute tables (DSM covariates), TOON offers significant token savings over JSON.  
4. **Watch the LSM Space:** The transition from "LLM \+ GIS Tools" to "Native Large Spatial Models" is the most significant trend to watch in late 2025/2026. These models will likely render current serialization debates obsolete by operating directly on spatial embeddings.

This architecture provides a robust, scalable, and hallucination-resistant foundation for the next generation of AI-powered geospatial analysis, specifically tailored to the demanding requirements of Digital Soil Mapping and multi-modal Earth Observation.

### ---

**Citations**

1

#### **Works cited**

1. geo\_from\_wkt() \- Kusto \- Microsoft Learn, accessed on January 7, 2026, [https://learn.microsoft.com/en-us/kusto/query/geo-from-wkt-function?view=microsoft-fabric](https://learn.microsoft.com/en-us/kusto/query/geo-from-wkt-function?view=microsoft-fabric)  
2. Random GeoJSON and WKT with randgeo \- rOpenSci, accessed on January 7, 2026, [https://ropensci.org/blog/2017/04/20/randgeo/](https://ropensci.org/blog/2017/04/20/randgeo/)  
3. Advancements in Visual Language Models for Remote Sensing: Datasets, Capabilities, and Enhancement Techniques \- arXiv, accessed on January 7, 2026, [https://arxiv.org/html/2410.17283v3](https://arxiv.org/html/2410.17283v3)  
4. Spectral Image Tokenizer \- CVF Open Access, accessed on January 7, 2026, [https://openaccess.thecvf.com/content/ICCV2025/papers/Esteves\_Spectral\_Image\_Tokenizer\_ICCV\_2025\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025/papers/Esteves_Spectral_Image_Tokenizer_ICCV_2025_paper.pdf)  
5. On the Use of LLMs for GIS-Based Spatial Analysis \- MDPI, accessed on January 7, 2026, [https://www.mdpi.com/2220-9964/14/10/401](https://www.mdpi.com/2220-9964/14/10/401)  
6. Foundation Models for Geospatial Reasoning: Assessing the Capabilities of Large Language Models in Understanding Geometries and Topological Spatial Relations \- arXiv, accessed on January 7, 2026, [https://arxiv.org/html/2505.17136v1](https://arxiv.org/html/2505.17136v1)  
7. Full article: GeoFM: how will geo-foundation models reshape spatial data science and GeoAI? \- Taylor & Francis Online, accessed on January 7, 2026, [https://www.tandfonline.com/doi/full/10.1080/13658816.2025.2543038](https://www.tandfonline.com/doi/full/10.1080/13658816.2025.2543038)  
8. Revolutionizing earth observation with geospatial foundation models on AWS, accessed on January 7, 2026, [https://aws.amazon.com/blogs/machine-learning/revolutionizing-earth-observation-with-geospatial-foundation-models-on-aws/](https://aws.amazon.com/blogs/machine-learning/revolutionizing-earth-observation-with-geospatial-foundation-models-on-aws/)  
9. Prithvi Model: Geospatial Foundation \- Emergent Mind, accessed on January 7, 2026, [https://www.emergentmind.com/topics/prithvi-model](https://www.emergentmind.com/topics/prithvi-model)  
10. Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications \- NASA Technical Reports Server, accessed on January 7, 2026, [https://ntrs.nasa.gov/api/citations/20240015391/downloads/RSE%20Prithvi%20Global.pdf?t](https://ntrs.nasa.gov/api/citations/20240015391/downloads/RSE%20Prithvi%20Global.pdf?t)  
11. Difference between WKT and GeoJson (data parsing) \- Stack Overflow, accessed on January 7, 2026, [https://stackoverflow.com/questions/38419640/difference-between-wkt-and-geojson-data-parsing](https://stackoverflow.com/questions/38419640/difference-between-wkt-and-geojson-data-parsing)  
12. GeoDS/GeoFM-TopologicalRelations: Foundation Models for Geospatial Reasoning: Assessing the Capabilities of Large Language Models in Understanding Geometries and Topological Spatial Relations \- GitHub, accessed on January 7, 2026, [https://github.com/GeoDS/GeoFM-TopologicalRelations](https://github.com/GeoDS/GeoFM-TopologicalRelations)  
13. GeoAnalystBench: A GeoAI benchmark for assessing large language models for spatial analysis workflow and code generation \- arXiv, accessed on January 7, 2026, [https://arxiv.org/html/2509.05881v1](https://arxiv.org/html/2509.05881v1)  
14. GeoAnalystBench: A GeoAI benchmark for assessing large language models for spatial analysis workflow and code generation \- GitHub, accessed on January 7, 2026, [https://github.com/GeoDS/GeoAnalystBench](https://github.com/GeoDS/GeoAnalystBench)  
15. Benchmarked JSON vs TOON for AI reasoners — 40–80% token savings. Real numbers inside. : r/LocalLLaMA \- Reddit, accessed on January 7, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1p0gzz9/benchmarked\_json\_vs\_toon\_for\_ai\_reasoners\_4080/](https://www.reddit.com/r/LocalLLaMA/comments/1p0gzz9/benchmarked_json_vs_toon_for_ai_reasoners_4080/)  
16. Is JSON Outdated? The Reasons Why the New LLM-Era Format "TOON" Saves Tokens, accessed on January 7, 2026, [https://dev.to/akari\_iku/is-json-outdated-the-reasons-why-the-new-llm-era-format-toon-saves-tokens-31e5](https://dev.to/akari_iku/is-json-outdated-the-reasons-why-the-new-llm-era-format-toon-saves-tokens-31e5)  
17. TOON vs JSON: A Token-Optimized Data Format for Reducing LLM Costs | Tensorlake, accessed on January 7, 2026, [https://www.tensorlake.ai/blog/toon-vs-json](https://www.tensorlake.ai/blog/toon-vs-json)  
18. Spatial Awareness in LLMs \- OpenReview, accessed on January 7, 2026, [https://openreview.net/pdf?id=nUCILW9uDD](https://openreview.net/pdf?id=nUCILW9uDD)  
19. Earth System Transformers \- Emergent Mind, accessed on January 7, 2026, [https://www.emergentmind.com/topics/earth-system-transformers](https://www.emergentmind.com/topics/earth-system-transformers)  
20. S2 \- H3, accessed on January 7, 2026, [https://h3geo.org/docs/comparisons/s2/](https://h3geo.org/docs/comparisons/s2/)  
21. Expanded AI Model with Global Data Enhances Earth Science Applications, accessed on January 7, 2026, [https://science.nasa.gov/science-research/ai-geospatial-model-earth/](https://science.nasa.gov/science-research/ai-geospatial-model-earth/)  
22. RS-RAG: Bridging Remote Sensing Imagery and Comprehensive Knowledge with a Multi-Modal Dataset and Retrieval-Augmented Generation Model \- arXiv, accessed on January 7, 2026, [https://arxiv.org/html/2504.04988v1](https://arxiv.org/html/2504.04988v1)  
23. Prithvi 100M · Models \- Dataloop, accessed on January 7, 2026, [https://dataloop.ai/library/model/ibm-nasa-geospatial\_prithvi-100m/](https://dataloop.ai/library/model/ibm-nasa-geospatial_prithvi-100m/)  
24. SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery \- Wageningen University & Research, accessed on January 7, 2026, [https://research.wur.nl/en/publications/satclip-global-general-purpose-location-embeddings-with-satellite/](https://research.wur.nl/en/publications/satclip-global-general-purpose-location-embeddings-with-satellite/)  
25. SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery | Request PDF \- ResearchGate, accessed on January 7, 2026, [https://www.researchgate.net/publication/390719221\_SatCLIP\_Global\_General-Purpose\_Location\_Embeddings\_with\_Satellite\_Imagery](https://www.researchgate.net/publication/390719221_SatCLIP_Global_General-Purpose_Location_Embeddings_with_Satellite_Imagery)  
26. SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery | Proceedings of the AAAI Conference on Artificial Intelligence, accessed on January 7, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/32457](https://ojs.aaai.org/index.php/AAAI/article/view/32457)  
27. Enhancing Satellite Data Interpretation through Soft Prompting in Embedding Space with Pre-trained Large Language Model, accessed on January 7, 2026, [https://agu.confex.com/agu/agu24/meetingapp.cgi/Paper/1747489](https://agu.confex.com/agu/agu24/meetingapp.cgi/Paper/1747489)  
28. Prompt Tuning with Soft Prompts, accessed on January 7, 2026, [https://learnprompting.org/docs/trainable/soft\_prompting](https://learnprompting.org/docs/trainable/soft_prompting)  
29. Using PostgreSQL as a vector database in RAG \- Azalio, accessed on January 7, 2026, [https://www.azalio.io/using-postgresql-as-a-vector-database-in-rag/](https://www.azalio.io/using-postgresql-as-a-vector-database-in-rag/)  
30. Pgvector and PostGIS: Unlocking Advanced PostgreSQL Use Cases with Vultr, accessed on January 7, 2026, [https://blogs.vultr.com/PG-Vector-PostGIS](https://blogs.vultr.com/PG-Vector-PostGIS)  
31. Building a job search engine with PostgreSQL's advanced search features \- AWS, accessed on January 7, 2026, [https://aws.amazon.com/blogs/database/building-a-job-search-engine-with-postgresqls-advanced-search-features/](https://aws.amazon.com/blogs/database/building-a-job-search-engine-with-postgresqls-advanced-search-features/)  
32. Postgres as a vector Database | Implementing Hybrid search with Postgres for RAG Using Groq. | by Meeran Malik | Medium, accessed on January 7, 2026, [https://medium.com/@meeran03/postgres-as-a-vector-database-implementing-hybrid-search-with-postgres-for-rag-using-groq-494ca3e41d57](https://medium.com/@meeran03/postgres-as-a-vector-database-implementing-hybrid-search-with-postgres-for-rag-using-groq-494ca3e41d57)  
33. Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots \- arXiv, accessed on January 7, 2026, [https://arxiv.org/html/2505.20288v1](https://arxiv.org/html/2505.20288v1)  
34. Hierarchical Geographic Tokenization \- Emergent Mind, accessed on January 7, 2026, [https://www.emergentmind.com/topics/hierarchical-geographic-item-tokenization](https://www.emergentmind.com/topics/hierarchical-geographic-item-tokenization)  
35. Build Your First GIS AI Agent by GIS MCP Server \+ LangChain | by Mahdi Nazari \- Medium, accessed on January 7, 2026, [https://medium.com/@mahdinazari75/build-your-first-gis-ai-agent-by-gis-mcp-server-langchain-c0c1bfa36f6d](https://medium.com/@mahdinazari75/build-your-first-gis-ai-agent-by-gis-mcp-server-langchain-c0c1bfa36f6d)  
36. Custom AI Agent Development | Agentic GIS Automation \- Geospatial Solutions, accessed on January 7, 2026, [https://geospatialsolutions.co/solutions/ai-agent-development](https://geospatialsolutions.co/solutions/ai-agent-development)  
37. Large Spatial Models (LSM) \- Geoinformation and Big Data Research Laboratory, accessed on January 7, 2026, [https://giscience.psu.edu/2025/05/02/building-the-large-spatial-models-lsm/](https://giscience.psu.edu/2025/05/02/building-the-large-spatial-models-lsm/)  
38. Prithvi WxC: Foundation Model for Weather and Climate \- arXiv, accessed on January 7, 2026, [https://arxiv.org/html/2409.13598v1](https://arxiv.org/html/2409.13598v1)  
39. Leveraging LLMs for advanced spatial decision support: The case of SATGPT \- AI for Good, accessed on January 7, 2026, [https://aiforgood.itu.int/event/leveraging-llms-for-advanced-spatial-decision-support-the-case-of-satgpt/](https://aiforgood.itu.int/event/leveraging-llms-for-advanced-spatial-decision-support-the-case-of-satgpt/)  
40. MapBot: A Multi-Modal Agent for Geospatial Analysis \- IFAAMAS, accessed on January 7, 2026, [https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p3059.pdf](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p3059.pdf)  
41. ToDo: Token Downsampling for Efficient Generation of High-Resolution Images \- IJCAI, accessed on January 7, 2026, [https://www.ijcai.org/proceedings/2024/1036.pdf](https://www.ijcai.org/proceedings/2024/1036.pdf)  
42. 5 Chunking Strategies for RAG: Optimize Your Retrieval-Augmented Generation Pipeline : r/NextGenAITool \- Reddit, accessed on January 7, 2026, [https://www.reddit.com/r/NextGenAITool/comments/1o3p5xv/5\_chunking\_strategies\_for\_rag\_optimize\_your/](https://www.reddit.com/r/NextGenAITool/comments/1o3p5xv/5_chunking_strategies_for_rag_optimize_your/)  
43. Building AI-Powered Search and RAG with PostgreSQL and Vector Embeddings \- Medium, accessed on January 7, 2026, [https://medium.com/@richardhightower/building-ai-powered-search-and-rag-with-postgresql-and-vector-embeddings-09af314dc2ff](https://medium.com/@richardhightower/building-ai-powered-search-and-rag-with-postgresql-and-vector-embeddings-09af314dc2ff)  
44. Supercharging vector search performance and relevance with pgvector 0.8.0 on Amazon Aurora PostgreSQL | AWS Database Blog, accessed on January 7, 2026, [https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql/](https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql/)  
45. Evaluating Intrinsic Geospatial Topological Reasoning in LLMs \- USC Viterbi, accessed on January 7, 2026, [https://viterbi-web.usc.edu/\~sabek/pdf/25\_workshop\_llm\_spatial\_benchmark.pdf](https://viterbi-web.usc.edu/~sabek/pdf/25_workshop_llm_spatial_benchmark.pdf)  
46. Spectral-spatial wave and frequency interactive transformer for hyperspectral image classification \- PMC \- PubMed Central, accessed on January 7, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12297294/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12297294/)  
47. Large Language Model-Driven Structured Output: A Comprehensive Benchmark and Spatial Data Generation Framework \- MDPI, accessed on January 7, 2026, [https://www.mdpi.com/2220-9964/13/11/405](https://www.mdpi.com/2220-9964/13/11/405)  
48. Semantic Tokenization-Based Mamba for Hyperspectral Image Classification \- IEEE Xplore, accessed on January 7, 2026, [https://ieeexplore.ieee.org/iel8/4609443/10766875/10838328.pdf](https://ieeexplore.ieee.org/iel8/4609443/10766875/10838328.pdf)  
49. ibm-nasa-geospatial/Prithvi-EO-2.0-300M · Hugging Face, accessed on January 7, 2026, [https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)  
50. GeoAnalystBench: A GeoAI benchmark for assessing large language models for spatial analysis workflow and code generation \- arXiv, accessed on January 7, 2026, [https://arxiv.org/pdf/2509.05881?](https://arxiv.org/pdf/2509.05881)  
51. General Geospatial Inference with a Population Dynamics Foundation Model \- arXiv, accessed on January 7, 2026, [https://arxiv.org/html/2411.07207v3](https://arxiv.org/html/2411.07207v3)  
52. SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery \- Microsoft, accessed on January 7, 2026, [https://www.microsoft.com/en-us/research/publication/satclip-global-general-purpose-location-embeddings-with-satellite-imagery/](https://www.microsoft.com/en-us/research/publication/satclip-global-general-purpose-location-embeddings-with-satellite-imagery/)  
53. IBM and NASA release a new version of Prithvi \- IBM Research, accessed on January 7, 2026, [https://research.ibm.com/blog/prithvi2-geospatial](https://research.ibm.com/blog/prithvi2-geospatial)  
54. I Introduction \- arXiv, accessed on January 7, 2026, [https://arxiv.org/html/2511.01082v1](https://arxiv.org/html/2511.01082v1)