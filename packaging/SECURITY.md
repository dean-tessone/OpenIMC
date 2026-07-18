# Desktop build security review

## Release security boundary

The release pipeline treats source, Python packages, model downloads, signing
credentials, and generated artifacts as separate trust boundaries.

- PyInstaller runs without API or signing credentials in its environment.
- CellSAM is fetched from one reviewed, full Git commit rather than a moving
  branch.
- PyInstaller and its hooks are exact-version build inputs; UPX and one-file
  extraction are disabled.
- GitHub Actions are pinned to full commit hashes and checkout does not retain
  repository credentials.
- `pip-audit` blocks known vulnerable Python dependency versions and produces a
  CycloneDX SBOM for the resolved environment, subject only to the documented
  Intel macOS PyTorch exception below.
- The generated app is recursively checked for credential filenames, exact
  build-time API tokens, and OpenAI-shaped key strings.
- Tagged Windows bundles are signed and RFC 3161 timestamped with Azure Artifact
  Signing. The main executable's Authenticode status must validate afterward.
- Tagged macOS bundles are Developer ID signed with hardened runtime, submitted
  to Apple's notarization service, stapled, and assessed before publication.
- Microsoft Defender scans Windows output; ClamAV scans Ubuntu output.
- The archive is created only after signing and scanning, then receives a
  SHA-256 checksum and GitHub artifact/SBOM attestation.

## Credential handling

`OPENAI_API_KEY` and `DEEPCELL_ACCESS_TOKEN` are application runtime inputs, not
compiled configuration. The GUI retains them in process memory for the current
session. The tagged functional test may receive `DEEPCELL_ACCESS_TOKEN` through
GitHub's masked secret environment, but the PyInstaller step cannot access it.
Azure OIDC credentials and Artifact Signing settings are scoped to signing
steps that run after the bundle is built.

Do not add keys to source, `.env` files, PyInstaller data files, test fixtures,
preferences, workflow command lines, or release archives.

## Residual risks

- Antivirus signatures and reputation systems change after release; a clean
  scan is evidence, not a permanent guarantee.
- GitHub-hosted Windows and Ubuntu scanners are two engines, not every vendor.
- Most scientific Python dependencies resolve within declared version ranges.
  Each build is audited and recorded in its SBOM, but it is not bit-for-bit
  reproducible across dates.
- CellSAM weights are downloaded at runtime from DeepCell and are outside the
  executable's code-signing boundary. CellSAM verifies its declared model hash;
  release validation must exercise the actual download and inference path.
- GitHub artifact attestations establish build provenance and integrity; they
  do not independently prove that software is safe.

### Intel macOS conda-forge PyTorch boundary

PyPI no longer publishes current Intel macOS PyTorch wheels. The Intel workflow
therefore uses the current conda-forge PyTorch 2.12.1 and torchvision 0.27.1
packages instead of freezing the application on PyPI's obsolete PyTorch 2.2.2.
PyTorch 2.12.1 has one known, low-severity local TorchScript advisory
(`GHSA-rrmf-rvhw-rf47`) whose fix is scheduled for PyTorch 2.13.0. The workflow
allowlists only that exact advisory on Intel macOS; any new advisory still
fails the build. Apple Silicon, Windows, and Ubuntu receive no exception.

The Intel build must only load the official model weights fetched by OpenIMC's
Cellpose and CellSAM integrations. Untrusted or user-supplied PyTorch model
files are outside this release's security boundary. OpenIMC's patched CellSAM
route does not compile models with `torch.jit.script`. Remove the exception as
soon as conda-forge publishes a fixed Intel package, or retire the Intel
artifact if it can no longer meet the release security policy.

## Tagged release checklist

1. Review dependency and build-tool updates, especially the pinned CellSAM
   commit.
2. Confirm GitHub environment protection and least-privilege Azure signing
   roles are enabled.
3. Push a `v*` tag and require all four platform jobs to succeed.
4. Verify the Windows Authenticode signer, timestamp, SHA-256 checksum, SBOM,
   GitHub attestation, and both macOS notarization results before publishing.
5. Submit the final archive to any additional antivirus services required by
   your institution. Never upload a private or embargoed scientific dataset.
