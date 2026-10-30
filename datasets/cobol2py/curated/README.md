# Curated Cobol2py records

This directory is the only source that RONIN-IA's dataset-sync workflow may import.

Records are not copied here automatically from Actions artifacts. A maintainer must:
1. inspect the GnuCOBOL evidence and exact stdout/return-code observations;
2. verify source licensing and redistribution rights;
3. check duplicate/program-family leakage and split assignments;
4. review model proposals independently and reject unsupported claims;
5. update `manifest.json` to an approved status and point it at a reviewed train/dev file.

Do not put holdout records, unreviewed model proposals, third-party source code with unclear rights, generated Python, or benchmark secrets here. Keep holdout in a separate protected evaluation process.

Until the manifest is approved, this directory intentionally contains no training records.
