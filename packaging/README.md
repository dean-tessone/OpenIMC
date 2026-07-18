# OpenIMC desktop releases

Desktop releases are self-contained, directory-based PyInstaller bundles. End
users do not need Python, Git, a virtual environment, or administrator access.

## Build locally

Build on each operating system that you plan to support; PyInstaller does not
cross-compile. Python 3.12 is the current release-build baseline.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip install -r requirements-build.txt
python scripts/build_desktop.py --version 0.1.0 --functional-test --archive
```

The script performs an offscreen smoke test of Qt and the major scientific
dependencies. `--functional-test` is the release gate: it runs real OME-TIFF
loading, CellSAM, Cellpose, watershed, feature extraction, Combat, Harmony,
clustering, spatial graph construction, and data/state export through the
packaged executable. Its checkpointed report is written to
`build/functional-validation/openimc-functional-validation.json`.

Outputs are written to `dist/`:

- macOS: `OpenIMC.app` and, with `--archive`, a `.dmg`
- Windows: `OpenIMC/OpenIMC.exe` and a `.zip`
- Linux: `OpenIMC/OpenIMC` and a `.tar.gz`

Use `--console` for a diagnostic build that shows Python output. Use
`--skip-smoke-test` only while investigating a build failure. The functional
test uses the repository fixture and locally cached CellSAM/Cellpose weights;
make sure the release machine has downloaded the selected models first.

## Important release constraints

- Build separately for Windows, Linux, macOS Intel, and macOS Apple Silicon.
- Linux bundles are compatible only with systems whose glibc is at least as new
  as the build host's; build on the oldest Linux distribution you support.
- macOS distribution outside the development machine requires Developer ID
  signing and Apple notarization. Set `OPENIMC_CODESIGN_IDENTITY` while building,
  then sign/notarize the final DMG in the release pipeline.
- Windows releases should be code-signed before publication to reduce
  SmartScreen warnings.
- Ilastik remains a separate application. OpenIMC locates and invokes it at
  runtime; it is not part of this Python bundle.
- Segmentation model weights may still be downloaded by Cellpose or CellSAM on
  first use. If fully offline operation is required, add the chosen weight files
  to the bundle and configure their runtime cache paths.
- API credentials are never build inputs or bundle data. The build removes
  `DEEPCELL_ACCESS_TOKEN` and `OPENAI_API_KEY` from child-process environments
  and scans the finished application for credential values and key-shaped
  strings. GUI-entered DeepCell and OpenAI keys are held in memory for the
  current session only and are not included in saved OpenIMC state.

## Why this is an application folder, not one file

PyTorch, Qt, HDF5, igraph, and image codecs rely on native shared libraries.
Keeping those libraries in the app bundle avoids a long extraction on every
launch and is substantially more reliable than PyInstaller's one-file mode.
The macOS `.app` and Windows `.exe` remain normal double-click applications.

## Automated Ubuntu and Windows builds

`.github/workflows/desktop-builds.yml` builds an Ubuntu 22.04 tarball, a Windows
Server 2022 zip, an Apple Silicon DMG, and an Intel Mac DMG on pushes to `main`
or `codex/dev`, every `v*` tag, and on manual dispatch. The workflow pins every
GitHub Action to a full commit, audits Python dependencies, runs key-security
tests, smoke-tests the frozen app, scans the finished folder with ClamAV or
Microsoft Defender where applicable, generates a CycloneDX SBOM, writes a
SHA-256 checksum, and creates a GitHub artifact attestation.

Tagged releases additionally run the complete frozen functional suite. Add a
repository secret named `DEEPCELL_ACCESS_TOKEN` so the test can download
CellSAM weights into the runner's user cache. The token is exposed only to the
functional-test step, never the PyInstaller build step, and the finished bundle
is scanned for its exact value before it can be archived.

Tagged Windows builds fail unless Azure Artifact Signing is configured. Create
an Artifact Signing account and certificate profile, grant the GitHub OIDC
identity the required signing role, then configure:

- secrets: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
- variables: `AZURE_ARTIFACT_SIGNING_ENDPOINT`,
  `AZURE_ARTIFACT_SIGNING_ACCOUNT`, `AZURE_ARTIFACT_SIGNING_PROFILE`

Branch and manually dispatched Windows builds are allowed to remain unsigned
so contributors can exercise the pipeline. Do not distribute those CI builds
as releases.

Tagged macOS builds fail unless Developer ID signing and Apple notarization are
configured. Add these repository secrets:

- `MACOS_CERTIFICATE_P12`: base64-encoded Developer ID Application certificate
- `MACOS_CERTIFICATE_PASSWORD`: password for the exported P12
- `MACOS_SIGNING_IDENTITY`: full Developer ID Application identity
- `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_SPECIFIC_PASSWORD`: notarization
  credentials

The certificate is imported into an ephemeral runner keychain after the
unsigned build and tests complete. The final DMG is notarized, stapled, and
assessed before its checksum is regenerated.

When all four tagged jobs pass, the workflow verifies their checksums and
attaches the archives, checksums, and SBOMs to the GitHub Release for the tag.
The stable public download page is:

`https://github.com/dean-tessone/OpenIMC/releases/latest`

## Antivirus and reputation expectations

The pipeline prevents publication when its configured scanners detect malware,
but no producer can guarantee that every current or future antivirus engine
will accept a new binary. Microsoft SmartScreen also evaluates download and
publisher reputation, not just whether a file is malicious or signed. A newly
signed file can therefore still show a reputation warning. For Windows, use a
consistent Artifact Signing identity for every release; Microsoft Store
distribution is the most reliable way to avoid SmartScreen's unknown-app
warning.

See `packaging/SECURITY.md` for the security boundary, residual risks, and the
release checklist.
