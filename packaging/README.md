# OpenIMC desktop releases

Desktop releases are self-contained, directory-based PyInstaller bundles. End
users do not need Python, Git, or a virtual environment. The macOS PKG installs
for every user and can request administrator authorization; users without it
can use the DMG and place OpenIMC in their personal `~/Applications` folder.

## Build locally

Build on each operating system that you plan to support; PyInstaller does not
cross-compile. Python 3.12 is the current release-build baseline. The automated
Intel macOS job obtains current PyTorch and torchvision packages from
conda-forge because PyPI no longer publishes current Intel macOS wheels.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip install -r requirements-build.txt
python scripts/build_desktop.py --version 0.1.0 --functional-test --archive
```

The script performs an offscreen smoke test of Qt and the major scientific
dependencies. `--functional-test` is the release gate: it runs real OME-TIFF
loading, Cellpose, watershed, feature extraction, Combat, Harmony, clustering,
spatial graph construction, and data/state export through the packaged
executable. If `DEEPCELL_ACCESS_TOKEN` is present and
`--allow-cellsam-download` is explicitly supplied, it additionally runs a live
CellSAM download and inference check. Its checkpointed report is written to
`build/functional-validation/openimc-functional-validation.json`.

Outputs are written to `dist/`:

- macOS: `OpenIMC.app` and, with `--archive`, a guided `.pkg` Installer plus a
  drag-to-Applications `.dmg`
- Windows: `OpenIMC/OpenIMC.exe` and a `.zip`
- Linux: `OpenIMC/OpenIMC`, a double-clickable `.deb` Installer, and a portable
  `.tar.gz`

Use `--console` for a diagnostic build that shows Python output. Use
`--skip-smoke-test` only while investigating a build failure. The functional
test uses the repository fixture and locally cached Cellpose weights. CellSAM
is reported as skipped unless its optional credentialed check is enabled.

## Runtime files

Installed application bundles are read-only. OpenIMC therefore stores its
methods log in a per-user writable location by default:

- macOS: `~/Library/Application Support/OpenIMC/logs/methods_log.jsonl`
- Windows: `%LOCALAPPDATA%\\OpenIMC\\logs\\methods_log.jsonl`
- Linux: `$XDG_STATE_HOME/openimc/logs/methods_log.jsonl`, or
  `~/.local/state/openimc/logs/methods_log.jsonl` when `XDG_STATE_HOME` is unset

Users can choose a different persistent file from **File → Methods Log File…**.
For scripted launches, `OPENIMC_LOG_FILE` overrides the default when no saved
GUI preference exists. No runtime file is written inside the executable or
application bundle.

## Important release constraints

- Build separately for Windows, Linux, macOS Intel, and macOS Apple Silicon.
- Linux bundles are compatible only with systems whose glibc is at least as new
  as the build host's; build on the oldest Linux distribution you support.
- macOS distribution outside the development machine requires Developer ID
  signing and Apple notarization. Set `OPENIMC_CODESIGN_IDENTITY` for the app
  and `OPENIMC_INSTALLER_SIGNING_IDENTITY` for the Installer package, then
  notarize and staple both the DMG and PKG in the release pipeline.
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
- The Windows bundle uses a project PyInstaller runtime hook to initialize
  PyTorch before Qt. This ordering is required by current Windows PyTorch and
  keeps the frozen Cellpose and CellSAM paths available.

## Why this is an application folder, not one file

PyTorch, Qt, HDF5, igraph, and image codecs rely on native shared libraries.
Keeping those libraries in the app bundle avoids a long extraction on every
launch and is substantially more reliable than PyInstaller's one-file mode.
The macOS `.app` and Windows `.exe` remain normal double-click applications.

## Automated Ubuntu and Windows builds

`.github/workflows/desktop-builds.yml` builds an Ubuntu 22.04 DEB Installer and
portable tarball, a Windows Server 2022 zip, and both PKG and DMG distributions
for Apple Silicon and Intel Macs on pushes to `main` or `codex/dev`, every `v*`
tag, and on manual dispatch. The workflow pins every GitHub Action to a full
commit, audits Python dependencies, runs packaging and key-security tests,
smoke-tests the frozen app, scans the finished folder with ClamAV or Microsoft
Defender where applicable, generates a CycloneDX SBOM, writes SHA-256
checksums, and creates GitHub artifact attestations for both archives and
installers.

Tagged releases additionally run the frozen functional suite. A DeepCell token
is not required to build or publish OpenIMC. If an optional repository secret
named `DEEPCELL_ACCESS_TOKEN` is configured, the same test also downloads the
CellSAM weights into the runner's user cache and runs live inference. The token
is exposed only to that functional-test process, never the PyInstaller build
step, and the finished bundle is scanned for its exact value before it can be
archived. End users provide their own DeepCell token in the application when
they choose CellSAM.

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
- `MACOS_INSTALLER_CERTIFICATE_P12`: base64-encoded Developer ID Installer certificate
- `MACOS_INSTALLER_CERTIFICATE_PASSWORD`: password for the Installer P12
- `MACOS_INSTALLER_SIGNING_IDENTITY`: full Developer ID Installer identity
- `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_SPECIFIC_PASSWORD`: notarization
  credentials

The certificates are imported into an ephemeral runner keychain after the
unsigned build and tests complete. The final DMG and PKG are notarized, stapled,
and assessed before their checksums are regenerated. Branch builds also produce
an unsigned PKG for testing, but it can receive a stronger Gatekeeper warning;
the signed and notarized PKG is the public installer.

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
