# LUCERA  
**Land-Use Classification Enhancement & Reliability Architecture**  
*A reproducible framework for spatial and temporal densification of land-cover labels, verification pipelines, and blockchain-anchored provenance.*

## Overview  
LUCERA is an open framework for constructing **high-resolution, high-frequency land-use / land-cover (LULC) datasets** through unified processing of multi-source Earth-observation data.  
The system integrates **AI-based super-resolution**, **crowdsourced in-situ verification**, and **traceable provenance** using the **Cardano blockchain**.

The framework was originally developed for the **EU-wide densification of CORINE Land Cover**, expanding its temporal refresh rate from multi-year cycles to **continuous update cadence**, and its spatial resolution down to the **10–1 m scale** through model-based inference and super-resolved Sentinel-2 imagery.

The same architecture generalises to global scale, enabling shared rules for dataset versioning, cross-validation, and reliability scoring.

---

## Core Objectives

- Produce **spatially densified** LULC maps (10–1 m equivalent) using Sentinel-2, super-resolved S2DR3/S2DR4 mosaics, and auxiliary datasets.  
- Build **temporally densified** label series by integrating continuous satellite streams with incremental updates.  
- Provide **verification tools** for field-level validation through mobile collection, expert review, and crowdsourcing.  
- Generate **provenance-anchored artefacts** stored immutably on Cardano (metadata only; no raw EO data on-chain).  
- Offer a complete **open pipeline** for data ingestion, model inference, validation, and export.

---

## System Architecture

### 1. Data Sources  
- Sentinel-2 L1C/L2A  
- Super-resolved S2DR3 (10×)  
- National catalogues (CORINE, Copernicus HRLs)  
- UAV / aerial imagery (optional)  
- Crowdsourced verification samples  

### 2. Processing Pipeline  
1. **Tiling & H3 geo-indexing**  
2. **Feature extraction / embeddings**  
3. **Label propagation & densification**  
4. **AI-based super-resolution**  
5. **Classifier with uncertainty estimation**  
6. **Verification loop**  
7. **Publishing + blockchain provenance anchoring**
