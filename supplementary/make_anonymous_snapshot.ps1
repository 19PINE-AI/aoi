param(
    [string]$OutputPath = "paper_aaai/anonymous_code_data_appendix.zip"
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$resolvedOutput = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$stageParent = Join-Path $tempRoot ("aoi-anonymous-" + [guid]::NewGuid().ToString("N"))
$stageRoot = Join-Path $stageParent "anonymous_code_data_appendix"
$fixedTime = [datetime]::SpecifyKind([datetime]"2000-01-01T00:00:00", [DateTimeKind]::Utc)

function Get-RelativeChildPath {
    param(
        [string]$Parent,
        [string]$Child
    )

    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $childFull = [IO.Path]::GetFullPath($Child)
    if (-not $childFull.StartsWith($parentFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside expected parent: $childFull"
    }
    return $childFull.Substring($parentFull.Length)
}

function Copy-RepoTree {
    param([string]$RelativePath)

    $sourceRoot = Join-Path $repoRoot $RelativePath
    Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | ForEach-Object {
        $relativeFile = Get-RelativeChildPath -Parent $repoRoot -Child $_.FullName
        $segments = $relativeFile -split '[\\/]'
        if ($segments -contains "_archive" -or
            $segments -contains "__pycache__" -or
            $_.Extension -in @(".pyc", ".pyo", ".log", ".tmp")) {
            return
        }

        $destination = Join-Path $stageRoot $relativeFile
        $destinationDirectory = Split-Path -Parent $destination
        New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $destination
    }
}

function Assert-AnonymousText {
    $identityPatterns = @(
        'Bojie\s+Li',
        'Li,\s*Bojie',
        'Noah\s+Shi',
        'Shi,\s*Noah',
        'Pine\s+AI',
        'University\s+of\s+Washington',
        '2606\.29472',
        '01\.me/research/aoi',
        'Agent-Computer Observation Interfaces Enable Dynamic Computer Use',
        'li2026aoi'
    )
    $secretPatterns = @(
        'sk-[A-Za-z0-9_-]{20,}',
        'AIza[0-9A-Za-z_-]{30,}',
        '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'
    )

    $textExtensions = @(".py", ".md", ".txt", ".json", ".html", ".js", ".css", ".ps1", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".example", ".cast")
    $textFiles = Get-ChildItem -LiteralPath $stageRoot -Recurse -File | Where-Object {
        $_.Name -eq ".env.example" -or $_.Extension -in $textExtensions
    }

    foreach ($file in $textFiles) {
        $content = Get-Content -LiteralPath $file.FullName -Raw
        foreach ($pattern in $identityPatterns + $secretPatterns) {
            if ($content -match $pattern) {
                throw "Anonymity or secret scan failed for $($file.FullName): pattern $pattern"
            }
        }
    }
}

try {
    New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null

    foreach ($directory in @("aoi", "benchmark_env", "dynacubench", "experiments", "results", "tests")) {
        Copy-RepoTree -RelativePath $directory
    }

    foreach ($file in @("requirements.txt", ".env.example")) {
        Copy-Item -LiteralPath (Join-Path $repoRoot $file) -Destination (Join-Path $stageRoot $file)
    }

    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "README_ANONYMOUS.md") -Destination (Join-Path $stageRoot "README.md")
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "ANONYMOUS_LICENSE.txt") -Destination (Join-Path $stageRoot "ANONYMOUS_LICENSE.txt")
    Copy-Item -LiteralPath (Join-Path $repoRoot "README.md") -Destination (Join-Path $stageRoot "PROJECT_README.md")

    $projectReadmePath = Join-Path $stageRoot "PROJECT_README.md"
    $projectReadme = Get-Content -LiteralPath $projectReadmePath -Raw
    $projectReadme = $projectReadme -replace 'Authors:\s*\*\*Bojie Li\*\* \(Pine AI\) and \*\*Noah Shi\*\* \(University of Washington\)\.', 'Authors: **Anonymous for peer review**.'
    $projectReadme = $projectReadme -replace 'Bojie Li', 'Anonymous Author'
    $projectReadme = $projectReadme -replace 'Noah Shi', 'Anonymous Author'
    $projectReadme = $projectReadme -replace 'Li, Bojie and Shi, Noah', 'Anonymous Authors'
    $projectReadme = $projectReadme -replace '\[arXiv:2606\.29472\]\(https://arxiv\.org/abs/2606\.29472\)', 'paper withheld during anonymous review'
    $projectReadme = $projectReadme -replace 'https://01\.me/research/aoi', 'website withheld during anonymous review'
    $projectReadme = $projectReadme -replace 'https://arxiv\.org/abs/2606\.29472', 'ANONYMOUS_PAPER_URL'
    $projectReadme = $projectReadme -replace 'arXiv:2606\.29472', 'anonymous manuscript'
    $projectReadme = $projectReadme -replace '2606\.29472', 'WITHHELD'
    $projectReadme = $projectReadme -replace 'Agent-Computer Observation Interfaces Enable Dynamic Computer Use', 'Anonymous Manuscript'
    $projectReadme = $projectReadme -replace 'li2026aoi', 'anonymous2026artifact'
    $projectReadme = $projectReadme -replace '\[`LICENSE`\]\(LICENSE\)', '[`ANONYMOUS_LICENSE.txt`](ANONYMOUS_LICENSE.txt)'
    $projectReadme = $projectReadme -replace '\[`LICENSE-DATA`\]\(LICENSE-DATA\)', '[`ANONYMOUS_LICENSE.txt`](ANONYMOUS_LICENSE.txt)'
    # Keep the repository README complete in the repository, but remove links and
    # commands for trees that are not present in the review archive.
    $projectReadme = $projectReadme -replace '(?m)^.*\*\*Website:\*\*.*\r?\n', ''
    $projectReadme = $projectReadme -replace '(?m)^\| \[`(?:paper|website|docs)/`\].*\r?\n', ''
    $projectReadme = $projectReadme -replace '(?m)^python website/scripts/build_data\.py.*\r?\n', ''
    $projectReadme = $projectReadme -replace '(?m)^See \[`docs/benchmark_design\.md`\].*\r?\n', ''
    $projectReadme = $projectReadme -replace '`tests/`, `website/`', '`tests/`'
    Set-Content -LiteralPath $projectReadmePath -Value $projectReadme -Encoding UTF8

    $experimentsReadmePath = Join-Path $stageRoot 'experiments/README.md'
    $experimentsReadme = Get-Content -LiteralPath $experimentsReadmePath -Raw
    $experimentsReadme = $experimentsReadme -replace '(?ms)\r?\n## Author-workflow & one-off runners — `_archive/`.*\z', ''
    Set-Content -LiteralPath $experimentsReadmePath -Value $experimentsReadme -Encoding UTF8

    $resultsReadmePath = Join-Path $stageRoot 'results/README.md'
    $resultsReadme = Get-Content -LiteralPath $resultsReadmePath -Raw
    $resultsReadme = $resultsReadme -replace ' and on the website', ''
    $resultsReadme = $resultsReadme -replace 'paper table / figure / website block', 'paper table / figure'
    $resultsReadme = $resultsReadme -replace '(?m)^\| `_archive/`.*\r?\n', ''
    $resultsReadme = $resultsReadme -replace '(?m)^Generated/consumed by `website/scripts/build_data\.py` and\r?\n', 'Generated/consumed by '
    $resultsReadme = $resultsReadme -replace '(?m)^python website/scripts/build_data\.py.*\r?\n', ''
    Set-Content -LiteralPath $resultsReadmePath -Value $resultsReadme -Encoding UTF8

    Assert-AnonymousText

    $manifestPath = Join-Path $stageRoot "FILE_MANIFEST_SHA256.txt"
    $manifestLines = Get-ChildItem -LiteralPath $stageRoot -Recurse -File |
        Where-Object { $_.FullName -ne $manifestPath } |
        Sort-Object FullName |
        ForEach-Object {
            $relative = (Get-RelativeChildPath -Parent $stageRoot -Child $_.FullName).Replace('\', '/')
            $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$hash  $relative"
        }
    Set-Content -LiteralPath $manifestPath -Value $manifestLines -Encoding ascii

    Get-ChildItem -LiteralPath $stageParent -Recurse -Force | ForEach-Object {
        $_.CreationTimeUtc = $fixedTime
        $_.LastAccessTimeUtc = $fixedTime
        $_.LastWriteTimeUtc = $fixedTime
    }

    $outputDirectory = Split-Path -Parent $resolvedOutput
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    if (Test-Path -LiteralPath $resolvedOutput) {
        Remove-Item -LiteralPath $resolvedOutput -Force
    }
    Push-Location $stageParent
    try {
        & tar.exe -a -c -f $resolvedOutput (Split-Path -Leaf $stageRoot)
        if ($LASTEXITCODE -ne 0) {
            throw "tar.exe failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    $zipHash = (Get-FileHash -LiteralPath $resolvedOutput -Algorithm SHA256).Hash.ToLowerInvariant()
    $zipSize = (Get-Item -LiteralPath $resolvedOutput).Length
    Write-Output "Created: $resolvedOutput"
    Write-Output "Bytes: $zipSize"
    Write-Output "SHA256: $zipHash"
} finally {
    $resolvedStageParent = [IO.Path]::GetFullPath($stageParent)
    if ($resolvedStageParent.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedStageParent).StartsWith("aoi-anonymous-")) {
        Remove-Item -LiteralPath $resolvedStageParent -Recurse -Force -ErrorAction SilentlyContinue
    }
}
