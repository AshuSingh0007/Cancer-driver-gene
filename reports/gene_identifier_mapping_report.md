# Gene identifier mapping report

> **Follow-up status:** The initial report below described only the v12.0 aliases then present. A matching human STRING v12.5 aliases file has since been acquired and validated. Current v12.5 results and the unresolved Ensembl crosswalk are recorded in the follow-up section at the end; those results supersede the earlier STRING-release limitation.

**Validation run:** `data/processed/validation_20261028/`  
**Sources:** TCGA GENCODE v36 STAR counts, NCG annotation tables, STRING links v12.5, local STRING aliases v12.0, COSMIC CGC v104.

## Verified mappings

- All 1,228 expression files are present in the local GDC manifest by filename, and the manifest file ID equals the enclosing directory UUID for all 1,228. These UUIDs establish GDC **file identity**, not case or sample identity.
- Expression matrices contain 60,660 identical versioned GENCODE gene IDs in every file. A symbol-only crosswalk against the unique NCG symbols found 3,343 exact one-symbol-to-one-NCG-ID candidates. This is exported as `data/processed/validation_20261028/expression_symbol_ncg_candidate_crosswalk.csv`; it is a provisional symbol crosswalk, not an Ensembl-to-Entrez authority mapping. 1,343 expression symbols occur on more than one Ensembl gene ID, so symbol-only joins require ambiguity controls.
- The current STRING alias data identifies 19,197 protein IDs with an Entrez alias; 19,196 have exactly one distinct Entrez ID, and one protein is ambiguous. Of those unambiguous protein IDs, 19,185 occur in the links file. Mapping table: `data/processed/validation_20261028/string_protein_entrez_mapping_v12_0_aliases.csv`.
- Within the 3,347 NCG Entrez IDs, 3,325 have at least one unambiguous STRING protein mapping (99.34%). Multiple proteins may map to one Entrez gene; count is distinct genes.

## Limitations and resolution

The available aliases are STRING v12.0, despite their enclosing directory name, while links are v12.5. No v12.5 alias file exists in the project inventory. Therefore all STRING-to-Entrez results and the copied graph are explicitly provisional. No download was made. Obtain the matching v12.5 aliases file from the same source/release as the links, preserve its original metadata/checksum, then rerun `src/validate_data.py` and `src/export_mapping_outputs.py` before model construction.

There is no authoritative local GENCODE v36 Ensembl-to-Entrez mapping table. GENCODE's release history identifies v36 as Ensembl release 102, so a release-102 BioMart mapping is the planned reference; the 3,343 symbol matches remain a candidate bridge only. COSMIC provides HGNC-style symbols and Ensembl versions in synonyms but is not used here as a mapping authority or target source. The [GENCODE release history](https://www.gencodegenes.org/human/releases.html) provides the version link; the [Ensembl BioMart documentation](https://www.ensembl.org/info/data/biomart/how_to_use_biomart.html) describes the gene and external-reference attributes.

## Outputs

- Detailed STRING mapping: `data/processed/validation_20261028/string_protein_entrez_mapping_v12_0_aliases.csv`
- Provisional expression symbol crosswalk: `data/processed/validation_20261028/expression_symbol_ncg_candidate_crosswalk.csv`
- File-level GDC identity/checksum manifest: `data/processed/validation_20261028/tcga_sample_case_manifest.csv`
- Machine-readable mapping and graph statistics: `data/processed/validation_20261028/string_validation_summary.json`

## Follow-up: release-matched STRING and expression alignment status

The approved acquisition produced `data/reference/metadata_20261028_151433/9606.protein.aliases.v12.5.txt.gz` and the extracted `9606.protein.aliases.v12.5.txt`. The extracted table header is `#string_protein_id, alias, source`. Current release-matched statistics are in `data/processed/validation_20261028_152218/string_validation_summary.json`:

- 19,197 proteins have an Entrez alias; 19,196 are unambiguous and one is ambiguous.
- Of 19,649 linked proteins, 19,185 have an unambiguous Entrez mapping.
- 3,325 of 3,347 NCG Entrez genes have at least one unambiguous STRING protein mapping (99.34%).
- The NCG-induced score-700 graph contains 2,799 connected nodes, 24,956 edges, no self-loops, 27 components, and a largest component of 2,743 nodes.

The downloaded Ensembl release-102 BioMart endpoint returned an HTML archive landing page, not TSV data. The invalid response was preserved separately and not parsed as an identifier table. The project's gene expression feature table remains keyed by versioned Ensembl ID, while labels and graph nodes are keyed by Entrez. Therefore no authoritative stable-ID join is available yet. The exact-symbol crosswalk remains a candidate aid only; integrated expression/label training rows and gene splits were intentionally not emitted.

## Superseding resolution: NCBI exact Ensembl crosswalk

The Ensembl v102 BioMart response did not yield a table, but the user-authorized follow-up acquired official NCBI Gene `gene_info`. Exact Ensembl stable-ID `dbXrefs` were used to create `data/processed/identifier_mapping_20261028_155713/ensembl_v36_to_entrez_ncbi_gene_info.csv`; symbols were not used for mapping. See `identifier_mapping_summary.json` for all counts, ambiguity, and discordance examples. There are 36,327 uniquely mapped expression IDs, 251 ambiguous and 24,082 unmatched. For the NCG + expression + graph universe, 3,304 genes remain (254 positive, 333 explicit zero, 2,717 unknown); 43 NCG entries were excluded. Among these eligible genes, 36 have a current symbol mismatch, retained as a QC flag rather than a join criterion.

The integrated model inputs are now constructed and aligned. Full model execution and interpretation are in `reports/phase6_model_evaluation.md`; current reproducible files are in `data/processed/integrated_model_inputs_20261028_155809/` and `outputs/experiments/integrated_20261028_161519/`.
