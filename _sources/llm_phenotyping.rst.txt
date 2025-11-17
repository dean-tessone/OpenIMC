LLM-Based Phenotyping
======================

LLM-based phenotyping uses large language models to suggest cell type annotations for clusters based on marker expression statistics. This feature provides AI-assisted phenotype suggestions to help accelerate cell type annotation workflows.

.. warning::
   **LLM suggestions are purely suggestive and not definitive outcomes.** Always manually review and validate all LLM suggestions before using them in your analysis. The LLM may make errors, and its suggestions should be treated as starting points for manual annotation, not as ground truth.

Overview
--------

After clustering, identifying cell types (phenotyping) is a critical but time-consuming step. LLM-based phenotyping uses OpenAI's language models to analyze marker expression patterns and suggest likely cell types for each cluster.

**How it works:**

1. **Statistics Computation**: For each cluster, computes marker expression statistics:
   - Top-K markers with highest expression (positive markers)
   - Top-K markers with lowest expression (negative markers)
   - Z-scores, log fold-changes, AUROC, and other metrics

2. **LLM Analysis**: Sends marker statistics to the selected LLM model along with:
   - System prompt (fine vs. broad mode)
   - User context (tissue type, cancer type, etc.)
   - Feature mode (markers only, morphometrics only, or both)

3. **Suggestion Generation**: LLM returns 3 phenotype suggestions per cluster with:
   - Phenotype name
   - Confidence percentage
   - Rationale explaining the suggestion

4. **User Selection**: You review and select from the suggestions for each cluster

**Key Features:**

- **Multiple Model Options**: Choose from different models based on speed, accuracy, cost, and reasoning capability
- **Fine vs. Broad Mode**: Control the level of detail in phenotype suggestions
- **Context-Aware**: Provide tissue/cancer type context for better suggestions
- **Feature Selection**: Choose which features to use (markers, morphometrics, or both)
- **Caching**: Results are cached to avoid redundant API calls
- **Cost**: Approximately $0.10 (10 cents) per query

Model Selection
---------------

OpenIMC supports multiple OpenAI models, each with different characteristics:

**Available Models:**

1. **gpt-5.1** (default): Latest model with reasoning capabilities
   - **Speed**: Moderate (slower than smaller models)
   - **Accuracy**: Highest
   - **Cost**: Highest
   - **Reasoning**: Yes (configurable: none, low, medium, high)
   - **Use case**: Best for complex datasets requiring detailed analysis

2. **gpt-5**: Standard GPT-5 model
   - **Speed**: Moderate
   - **Accuracy**: High
   - **Cost**: High
   - **Reasoning**: No
   - **Use case**: Good balance of accuracy and speed

3. **gpt-5-mini**: Smaller, faster version
   - **Speed**: Fast
   - **Accuracy**: Good
   - **Cost**: Moderate
   - **Reasoning**: No
   - **Use case**: Faster suggestions with good accuracy

4. **gpt-5-nano**: Smallest, fastest version
   - **Speed**: Very fast
   - **Accuracy**: Moderate
   - **Cost**: Low
   - **Reasoning**: No
   - **Use case**: Quick suggestions for simple datasets

5. **gpt-4.1**: Previous generation model
   - **Speed**: Moderate
   - **Accuracy**: High
   - **Cost**: High
   - **Reasoning**: No
   - **Use case**: Alternative to GPT-5 if needed

**Reasoning Level (gpt-5.1 only):**

When using gpt-5.1, you can configure the reasoning effort:

- **none** (default): No additional reasoning, fastest
- **low**: Minimal reasoning, slightly slower
- **medium**: Moderate reasoning, slower but more thorough
- **high**: Maximum reasoning, slowest but most thorough

**Model Selection Guidelines:**

- **For speed**: Use ``gpt-5-nano`` or ``gpt-5-mini``
- **For accuracy**: Use ``gpt-5.1`` with medium/high reasoning
- **For cost efficiency**: Use ``gpt-5-nano`` or ``gpt-5-mini``
- **For complex datasets**: Use ``gpt-5.1`` with reasoning enabled
- **For simple datasets**: Use ``gpt-5-mini`` or ``gpt-5-nano``

**Cost Considerations:**

- Models are charged per token (input + output)
- Larger models (gpt-5.1, gpt-5) cost more per token
- Reasoning increases cost and time
- **Estimated cost per query**: Approximately $0.10 (10 cents) per cluster
  - Cost varies by model, reasoning level, and number of markers
  - Smaller models (gpt-5-nano, gpt-5-mini) may cost less
  - Larger models with reasoning may cost more
- Monitor your OpenAI account usage to avoid unexpected charges

Inputs
------

The LLM receives several inputs to generate phenotype suggestions:

**1. Marker Statistics (Computed Automatically):**

For each cluster, the following statistics are computed:

- **Top-K Positive Markers**: Markers with highest expression in the cluster
  - Z-score (across clusters)
  - Log fold-change (vs. other clusters)
  - AUROC (area under ROC curve)
  - Mean expression
  - Percent positive cells
  - Within-cluster distribution (min, mean, median, max)
  - Range across clusters

- **Top-K Negative Markers**: Markers with lowest expression in the cluster
  - Same statistics as positive markers

**2. Feature Mode:**

Controls which features are included in the analysis:

- **Markers only**: Uses only intensity features (marker expression)
  - Includes: ``_mean``, ``_median``, ``_std``, ``_mad``, ``_p10``, ``_p90``, ``_integrated``, ``_frac_pos``
  - Excludes: Morphometric features (area, perimeter, etc.)
  - Top-K intensity: Number of top markers to include (default: 5, range: 1-30)

- **Morphometrics only**: Uses only morphometric features
  - Includes: Area, perimeter, eccentricity, solidity, etc.
  - Excludes: Intensity features
  - Top-K morphometric: Number of top morphometric features to include (default: 5, range: 1-30)

- **Both** (default): Uses both markers and morphometrics
  - Includes both intensity and morphometric features
  - Top-K intensity: Number of top markers (default: 5)
  - Top-K morphometric: Number of top morphometric features (default: 5)

**3. Context (Optional):**

User-provided context about the dataset:

- **Cohort/tissue context**: Free-text description
  - Examples: "human colorectal cancer", "mouse liver tissue", "breast cancer TMA"
  - Helps LLM provide context-appropriate suggestions
  - Optional but recommended for better accuracy

**4. System Prompt Mode:**

Controls the level of detail in suggestions:

- **Fine cell types (detailed)** (default): Suggests specific cell types
  - Examples: "CD4+ T cells", "M1 Macrophages", "Epithelial cells"
  - More detailed but may be less confident
  - Better for well-characterized datasets

- **Broad cell types**: Suggests broad categories first
  - Examples: "Lymphoid", "Myeloid", "Stroma", "Tumor"
  - More confident but less specific
  - Can specify subtypes when confidence is high (>50%)
  - Better for exploratory analysis or uncertain datasets

**5. Normalization Context:**

Automatically detected from your feature extraction settings:

- If arcsinh transformation was used: "intensities are arcsinh-transformed"
- If raw values: "intensities are raw values"
- Helps LLM interpret marker expression values correctly

Outputs
-------

The LLM returns structured JSON with phenotype suggestions for each cluster:

**Output Structure:**

.. code-block:: json

   {
     "cluster_id": "1",
     "phenotype_guesses": [
       {
         "name": "CD4+ T cells",
         "rationale": "High expression of CD3, CD4, and CD45. Low expression of CD8 and CD20.",
         "confidence": 75.5
       },
       {
         "name": "Helper T cells",
         "rationale": "CD3+ CD4+ profile consistent with helper T cell lineage.",
         "confidence": 20.0
       },
       {
         "name": "T cells",
         "rationale": "General T cell markers present, but subtype uncertain.",
         "confidence": 4.5
       }
     ],
     "key_markers_positive": ["CD3", "CD4", "CD45"],
     "key_markers_negative": ["CD8", "CD20"],
     "notes": "Strong T cell signature with helper T cell characteristics."
   }

**Output Fields:**

1. **phenotype_guesses**: Array of 3 phenotype suggestions (ranked by confidence)
   - **name**: Suggested cell type name
   - **rationale**: Explanation of why this phenotype was suggested
   - **confidence**: Confidence percentage (0-100%, must sum to 100% across all 3)

2. **key_markers_positive**: List of markers that support this phenotype
   - Markers with high expression in this cluster

3. **key_markers_negative**: List of markers that argue against this phenotype
   - Markers with low expression in this cluster

4. **notes**: Additional observations or caveats

**Display in GUI:**

- Each cluster shows a radio button group with 3 phenotype options
- Options are ranked by confidence (highest first)
- Confidence percentages are displayed
- Rationale is shown below each option
- You can select any of the 3 options or manually edit

Fine vs. Broad Mode
-------------------

The system prompt mode controls whether the LLM suggests detailed or broad cell types.

**Fine Cell Types Mode (Default):**

- **Goal**: Suggest specific, detailed cell types
- **Examples**: 
  - "CD4+ T cells"
  - "M1 Macrophages"
  - "Epithelial cells"
  - "Endothelial cells"
- **Use when**:
  - You have a well-characterized panel
  - You need specific cell type annotations
  - You have high confidence in marker specificity
- **Advantages**:
  - More informative annotations
  - Better for publication figures
  - More actionable results
- **Disadvantages**:
  - May be less confident
  - Can suggest incorrect specific types if markers are ambiguous

**Broad Cell Types Mode:**

- **Goal**: Suggest broad categories first, with optional specificity
- **Examples**:
  - "Lymphoid" (or "T cells" if confidence >50%)
  - "Myeloid" (or "Macrophages" if confidence >50%)
  - "Stroma"
  - "Tumor"
- **Use when**:
  - You're doing exploratory analysis
  - Your panel has limited specificity
  - You want more conservative suggestions
- **Advantages**:
  - More confident suggestions
  - Less likely to be wrong
  - Good starting point for manual refinement
- **Disadvantages**:
  - Less informative
  - May need manual refinement to specific types

**Mode Selection Guidelines:**

- Start with **Broad mode** for exploratory analysis
- Switch to **Fine mode** once you understand your data better
- Use **Fine mode** for publication-ready annotations
- Use **Broad mode** if suggestions seem too specific or uncertain

Using LLM-Based Phenotyping in the GUI
---------------------------------------

1. **Complete Clustering**: Ensure clustering has been run and clusters are available

2. **Open Phenotype Annotation Dialog**: 
   - In the clustering dialog, click **"Annotate Phenotypes"** button
   - This opens the phenotype annotation dialog

3. **Open LLM Suggestion Dialog**:
   - Click **"Suggest phenotypes with LLM…"** button
   - This opens the LLM phenotyping dialog

4. **Configure LLM Settings**:
   - **OpenAI API Key**: Enter your OpenAI API key (required)
     - Get an API key from https://openai.com/
     - Key is masked for security
   - **Cohort/tissue context**: Enter optional context (e.g., "human colorectal cancer")
   - **Model**: Select model (default: gpt-5.1)
   - **Reasoning level**: Select reasoning level if using gpt-5.1 (default: none)
   - **System prompt**: Choose "Fine cell types" or "Broad cell types"
   - **Feature mode**: Choose "Markers only", "Morphometrics only", or "Both"
   - **Top-K intensity**: Number of top markers to include (default: 5)
   - **Top-K morphometric**: Number of top morphometric features (default: 5, only if "Both" mode)

5. **Run Suggestions**:
   - Click **"Run Suggestion"** button
   - Progress bar shows analysis progress
   - Results are cached to avoid redundant API calls

6. **Review Suggestions**:
   - For each cluster, review the 3 suggested phenotypes
   - Read the rationale for each suggestion
   - Check confidence percentages
   - Select the most appropriate option using radio buttons

7. **Apply Names**:
   - Click **"Apply Names"** button to apply selected phenotypes
   - Selected phenotypes are applied to clusters
   - You can still manually edit after applying

8. **Close Dialog**:
   - Click **"Close"** to return to clustering dialog
   - Cached results are preserved for future use

**Caching:**

- Results are automatically cached based on cluster IDs and settings
- If you re-run with the same clusters, cached results are displayed immediately
- Cache persists until you close the clustering dialog
- To refresh suggestions, re-run the analysis

Important Warnings and Limitations
-----------------------------------

.. warning::
   **LLM suggestions are not definitive.** Always manually review and validate all suggestions before using them in your analysis.

**Key Limitations:**

1. **Not Ground Truth**: LLM suggestions are AI-generated and may contain errors
   - Always verify against known marker profiles
   - Cross-reference with literature or expert knowledge
   - Use suggestions as starting points, not final answers

2. **Model Limitations**: Language models have limitations
   - May not recognize novel or rare cell types
   - May misinterpret ambiguous marker profiles
   - May suggest incorrect phenotypes if markers are non-specific
   - Performance varies by model and dataset

3. **Cost Considerations**: API calls incur costs
   - **Estimated cost**: Approximately $0.10 (10 cents)  query
   - Models are charged per token (input + output)
   - Larger models and reasoning increase costs
   - Monitor your OpenAI account usage
   - Consider using smaller models for initial exploration

4. **Data Quality Dependency**: Suggestions depend on data quality
   - Poor clustering leads to poor suggestions
   - Missing or incorrect markers affect accuracy
   - Normalization artifacts can mislead the LLM

5. **Context Dependency**: Suggestions are context-dependent
   - Provide accurate tissue/cancer type context
   - Context helps but doesn't guarantee accuracy
   - Different contexts may yield different suggestions

**Best Practices:**

1. **Always Validate**: Never use LLM suggestions without manual review
2. **Start Broad**: Use broad mode for initial exploration
3. **Refine Manually**: Use suggestions as starting points for manual annotation
4. **Check Marker Profiles**: Verify suggestions against known marker profiles
5. **Use Multiple Models**: Try different models if suggestions seem off
6. **Provide Context**: Always provide accurate tissue/cancer type context
7. **Monitor Costs**: Keep track of API usage to avoid unexpected charges
8. **Cache Results**: Results are cached automatically to save costs

**When to Use LLM Phenotyping:**

- ✅ **Good for**: Initial exploration, getting started, brainstorming
- ✅ **Good for**: Well-characterized panels with known markers
- ✅ **Good for**: Standard cell types (T cells, B cells, macrophages, etc.)
- ❌ **Not ideal for**: Novel or rare cell types
- ❌ **Not ideal for**: Datasets with poor clustering quality
- ❌ **Not ideal for**: Panels with non-specific markers

**When NOT to Use LLM Phenotyping:**

- If you don't have an OpenAI API key
- If you need definitive, publication-ready annotations immediately
- If your dataset has very novel or rare cell types
- If cost is a major concern and you have many clusters

Tips and Best Practices
------------------------

1. **Model Selection**:
   - Start with ``gpt-5-mini`` for quick exploration
   - Use ``gpt-5.1`` with reasoning for complex datasets
   - Use ``gpt-5-nano`` for cost-sensitive workflows

2. **Mode Selection**:
   - Use **Broad mode** for initial exploration
   - Switch to **Fine mode** once you understand your data
   - Fine mode is better for publication figures

3. **Feature Selection**:
   - Use **"Both"** mode to include all available information
   - Use **"Markers only"** if morphometrics are not informative
   - Adjust Top-K values based on your panel size

4. **Context Provision**:
   - Always provide accurate tissue/cancer type context
   - Include relevant treatment information if applicable
   - More context = better suggestions

5. **Validation Workflow**:
   - Review all suggestions before applying
   - Check marker profiles in heatmaps/differential expression plots
   - Compare suggestions across different models
   - Manually correct incorrect suggestions

6. **Cost Management**:
   - **Estimated cost**: ~$0.10 per query
   - Use smaller models for initial exploration (may reduce cost)
   - Cache results to avoid redundant API calls
   - Monitor your OpenAI account usage

7. **Iterative Refinement**:
   - Start with broad suggestions
   - Refine manually based on marker profiles
   - Re-run with fine mode for final annotations
   - Combine LLM suggestions with manual annotation

8. **Quality Control**:
   - Verify suggestions make biological sense
   - Check that marker profiles match suggested phenotypes
   - Look for consistency across similar clusters
   - Flag uncertain suggestions for manual review

Example Workflow
----------------

1. **Initial Exploration**:
   - Run clustering
   - Open LLM phenotyping dialog
   - Select ``gpt-5-mini`` model
   - Choose "Broad cell types" mode
   - Provide tissue context
   - Run suggestions

2. **Review Suggestions**:
   - Review all cluster suggestions
   - Check confidence levels
   - Read rationales
   - Identify clusters needing manual review

3. **Refinement**:
   - Apply broad suggestions
   - Manually refine uncertain clusters
   - Re-run with "Fine cell types" mode for specific clusters
   - Use ``gpt-5.1`` with reasoning for difficult cases

4. **Final Annotation**:
   - Combine LLM suggestions with manual annotations
   - Verify all annotations in heatmaps/UMAPs
   - Export final annotations

Troubleshooting
---------------

**Common Issues:**

1. **"API Key Required" Error**:
   - Ensure you've entered a valid OpenAI API key
   - Check that the key starts with "sk-"
   - Verify your account has credits

2. **"No Clusters" Error**:
   - Run clustering first before using LLM phenotyping
   - Ensure clusters are available in the clustering dialog

3. **Poor Suggestions**:
   - Try a different model (e.g., gpt-5.1 with reasoning)
   - Switch between fine and broad mode
   - Provide more context
   - Check that clustering quality is good
   - Verify marker panel is appropriate

4. **High Costs**:
   - Use smaller models (gpt-5-nano, gpt-5-mini)
   - Disable reasoning for gpt-5.1
   - Reduce Top-K values
   - Cache results to avoid redundant calls

5. **Slow Performance**:
   - Use faster models (gpt-5-nano, gpt-5-mini)
   - Disable reasoning
   - Process clusters in batches
   - Check internet connection

6. **Invalid JSON Errors**:
   - The system automatically retries with repair instructions
   - If persistent, try a different model
   - Check that your API key has access to the selected model

**Getting Help:**

- Check OpenAI API status if experiencing connection issues
- Verify your API key permissions
- Review OpenAI documentation for model availability
- Check OpenIMC documentation for updates

