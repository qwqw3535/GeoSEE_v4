## GeoSEE v4: Geospatial Socio-Economic Estimation for the Visegrád Group

This repository contains the PyTorch implementation of **GeoSEE v4**, a framework for estimating socio-economic indicators from satellite imagery and auxiliary data across the Visegrád Group countries (Poland, Czech Republic, Slovakia, Hungary).

---

### Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Workflow Overview](#workflow-overview)

   - [Step 0: Preprocessing](#step-0-preprocessing)
   - [Step 1: Module Selection](#step-1-module-selection)
   - [Step 2: Paragraph Extraction](#step-2-paragraph-extraction)
   - [Step 3: LLM-based Prediction](#step-3-llm-based-prediction)

5. [Authors & Contact](#authors--contact)

---

## Prerequisites

- **Python**: 3.8.10
- **Packages**:

  ```
  numpy==1.24.4
  openai==0.28.0
  tqdm==4.66.1
  tensorflow==2.11.0
  arcgis==2.2.0.1
  rasterio==1.3.9
  shapely==2.0.2
  pillow==10.0.1
  torch (>=1.9)
  ```

---

## Configuration

- Open `config.ini` and set the following fields:

  - `api_key`: ArcGIS api Key
  - `ccode`: ISO country code of the target (e.g., `PL`, `CZ`, `SK`, `HU`). If you use ArcGIS service, ISO code is made of three alphabets (e.g., `POL`, `CZE`, `SVK`, `HUN`)
  - `zoom_level`, `timeline`: Resolution level, timeline of satellite imagery from ArcGIS world imagery
  - `threads`: CPU threads used when downloading satellite imagery

---

## Workflow Overview

### Step 0: Preprocessing

Prepare geographical boundaries and download satellite imagery for your target country.

1. **Generate sub-geography definitions:**

- Set `area_dict = True` and `create_geometry = True` in `preprocess.py`

  ```bash
  python preprocess.py
  ```

  This creates a sub-geography file under the `data/proxy` and `preprocessing` directory.

2. **Download satellite imagery:**

   - Update `create_image=True` in `preprocess.py`.
   - Re-run preprocessing:

     ```bash
     python preprocess.py --create_image True
     ```

     This downloads satellite imagery from ArcGIS server under `img_dir` in `config.ini`

3. **Download Nightlight imagery**

- Go to Earth Observation Group on Colorado School of Mines (https://eogdata.mines.edu/products/vnl/#annual_v2)
- Download target year nightlight GeoTIFF file
- Put it into `/resources/global_geotiff`

### Step 1: Module Selection

Interactively choose the model module best suited for your target variable.

```bash
jupyter notebook ask_gpt_about_module_selection.ipynb
```

- Provide a concise description and context for the target variable when prompted.

### Step 2: Paragraph Extraction

Convert numerical data into natural language paragraphs.

```bash
python extract_paragraph.py \
  --gpu \
  --ccode <COUNTRY_CODE> \
  --var <VARIABLE_NAME> \
  --adm <ADMIN_LEVEL> \
  --merged <MERGED_DATA_PATH>
```

- `--gpu`: Enable GPU acceleration.
- `--var`: Target socio-economic variable (e.g., `GRDP`, `population`).

### Step 3: LLM-based Prediction

Run the final model to predict the target indicator:

```bash
python main.py \
  --target_var <VARIABLE_NAME> \
  --gpt_version <GPT_MODEL>
```

- `--gpt_version`: Specify the OpenAI GPT version (e.g., `o4-mini`, `o3`).

---

## Authors & Contact

- **Donghyun Ahn**: [segaukwa@kaist.ac.kr](mailto:segaukwa@kaist.ac.kr)
- **Donggyu Lee**: [donggyu.lee@kaist.ac.kr](mailto:donggyu.lee@kaist.ac.kr)
- **Sumin Lee**: [dlsumn03@kaist.ac.kr](mailto:dlsumn03@kaist.ac.kr)
