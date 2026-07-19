param(
    [string]$OutputRoot = "C:\assets\checkpoints\a2c2_official",
    [string]$StatusPath = "C:\assets\checkpoints\a2c2_official\download_status.json"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$specs = @(
    [ordered]@{
        role = "base"
        repo_id = "k1000dai/smolvla_libero_spatial_scratch"
        revision = "caa0efcb24e261574c824366526c5775d3664cac"
        model_bytes = 906713328
        model_sha256 = "45F3B6FC1B8AE0B7CF3AB8EBD22336AB23EB3798A8BFEF027F5D45596C45A9BE"
    },
    [ordered]@{
        role = "prior"
        repo_id = "k1000dai/residual_transformer_libero_spatial_add_vlm_context"
        revision = "9c89cca4aae8eecc42a20084ef414ff74f94ba05"
        model_bytes = 123513140
        model_sha256 = "85D00523E8273A4141E288E4F6692224D50AAF8DF99AD8CCF7E72EE7BF3AB712"
    }
)

$files = @(".gitattributes", "README.md", "config.json", "model.safetensors", "train_config.json")
$started = Get-Date
$records = @()
$statusParent = Split-Path -Parent $StatusPath
New-Item -ItemType Directory -Force -Path $statusParent | Out-Null

function Write-Status([string]$State, $Current, $Exception) {
    $payload = [ordered]@{
        schema_version = 1
        state = $State
        pid = $PID
        started_utc = $started.ToUniversalTime().ToString("o")
        updated_utc = (Get-Date).ToUniversalTime().ToString("o")
        current = $Current
        records = $records
        exception = $Exception
    }
    $temporary = "$StatusPath.tmp"
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -Force -LiteralPath $temporary -Destination $StatusPath
}

try {
    Write-Status -State "running" -Current $null -Exception $null
    foreach ($spec in $specs) {
        $repoName = ($spec.repo_id -split "/")[-1]
        $destination = Join-Path (Join-Path $OutputRoot $repoName) $spec.revision
        New-Item -ItemType Directory -Force -Path $destination | Out-Null

        foreach ($file in $files) {
            $current = [ordered]@{role = $spec.role; repo_id = $spec.repo_id; revision = $spec.revision; file = $file}
            Write-Status -State "running" -Current $current -Exception $null
            $target = Join-Path $destination $file
            $partial = "$target.partial"
            $expectedHash = if ($file -eq "model.safetensors") { $spec.model_sha256 } else { $null }
            $expectedBytes = if ($file -eq "model.safetensors") { [int64]$spec.model_bytes } else { $null }

            $skip = $false
            if (Test-Path -LiteralPath $target) {
                $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToUpperInvariant()
                $actualBytes = (Get-Item -LiteralPath $target).Length
                $skip = ($null -eq $expectedHash) -or (($actualHash -eq $expectedHash) -and ($actualBytes -eq $expectedBytes))
                if (-not $skip) {
                    throw "Existing file fails frozen identity: $target bytes=$actualBytes sha256=$actualHash"
                }
            }

            if (-not $skip) {
                $encodedFile = $file.Replace(" ", "%20")
                $url = "https://huggingface.co/$($spec.repo_id)/resolve/$($spec.revision)/$encodedFile"
                $arguments = @(
                    "--location", "--fail", "--show-error", "--silent",
                    "--retry", "5", "--retry-delay", "2", "--continue-at", "-",
                    "--output", $partial, $url
                )
                & curl.exe @arguments
                if ($LASTEXITCODE -ne 0) {
                    throw "curl failed with exit code $LASTEXITCODE for $url"
                }
                Move-Item -Force -LiteralPath $partial -Destination $target
            }

            $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToUpperInvariant()
            $bytes = (Get-Item -LiteralPath $target).Length
            if (($null -ne $expectedHash) -and (($hash -ne $expectedHash) -or ($bytes -ne $expectedBytes))) {
                throw "Downloaded file fails frozen identity: $target bytes=$bytes sha256=$hash"
            }
            $records += [ordered]@{
                role = $spec.role
                repo_id = $spec.repo_id
                revision = $spec.revision
                file = $file
                path = $target
                bytes = $bytes
                sha256 = $hash
                skipped_existing_verified = $skip
            }
        }
    }
    Write-Status -State "completed" -Current $null -Exception $null
    exit 0
}
catch {
    $failure = [ordered]@{
        type = $_.Exception.GetType().Name
        message = $_.Exception.Message
        script_stack_trace = $_.ScriptStackTrace
    }
    Write-Status -State "failed" -Current $current -Exception $failure
    Write-Error $_
    exit 1
}
