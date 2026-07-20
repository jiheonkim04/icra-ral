[CmdletBinding()]
param(
  [switch]$Execute
)

$ErrorActionPreference = "Stop"
$Repo = "C:\Users\jiheo\tca_map"
$ManifestPath = Join-Path $Repo "reports\storage_cleanup\delete_manifest.json"
$LogPath = Join-Path $Repo "reports\storage_cleanup\deletion_execution_windows.json"
$Protected = @(
  "C:\Users\jiheo\tca_map\rollouts\2026_07_17",
  "C:\Users\jiheo\tca_map\rollouts\2026_07_18"
)

if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
  throw "Missing delete manifest: $ManifestPath"
}
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if (-not $Manifest.validation.passed) { throw "Delete manifest is not validated" }

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class StorageSize {
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  public static extern uint GetCompressedFileSizeW(string name, out uint high);
}
"@

function Get-AllocatedFileBytes([string]$Path) {
  [uint32]$high = 0
  [uint32]$low = [StorageSize]::GetCompressedFileSizeW($Path, [ref]$high)
  if ($low -eq [uint32]::MaxValue -and [Runtime.InteropServices.Marshal]::GetLastWin32Error() -ne 0) {
    return [int64](Get-Item -LiteralPath $Path -Force).Length
  }
  return ([int64]$high -shl 32) -bor [int64]$low
}

function Get-TargetMetrics([string]$Path) {
  $item = Get-Item -LiteralPath $Path -Force
  [int64]$apparent = 0
  [int64]$allocated = 0
  [int64]$files = 0
  $newest = $item.LastWriteTimeUtc
  $fileItems = if ($item.PSIsContainer) {
    Get-ChildItem -LiteralPath $item.FullName -Recurse -Force -File -ErrorAction Stop
  } else {
    @($item)
  }
  foreach ($file in $fileItems) {
    $apparent += [int64]$file.Length
    if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
      $allocated += [int64]$file.Length
    } else {
      $allocated += Get-AllocatedFileBytes $file.FullName
    }
    $files++
    if ($file.LastWriteTimeUtc -gt $newest) { $newest = $file.LastWriteTimeUtc }
  }
  [pscustomobject]@{
    apparent_size_bytes = $apparent
    allocated_size_bytes = $allocated
    file_count = $files
    newest_mtime_utc = $newest.ToString("o")
  }
}

function Test-ExclusiveRead([string]$Path) {
  $locked = [Collections.Generic.List[object]]::new()
  $item = Get-Item -LiteralPath $Path -Force
  $files = if ($item.PSIsContainer) {
    Get-ChildItem -LiteralPath $item.FullName -Recurse -Force -File -ErrorAction Stop
  } else { @($item) }
  foreach ($file in $files) {
    if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
    try {
      $stream = [IO.File]::Open($file.FullName, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
      $stream.Dispose()
    } catch {
      $locked.Add([pscustomobject]@{ path = $file.FullName; error = $_.Exception.Message })
    }
  }
  return $locked
}

$branch = git -C $Repo branch --show-current
$head = git -C $Repo rev-parse HEAD
$originHead = git -C $Repo rev-parse "origin/$branch"
$mergeBase = git -C $Repo merge-base HEAD "origin/$branch"
if ($head -ne $Manifest.source_git.local_head -or $originHead -ne $Manifest.source_git.origin_head -or $mergeBase -ne $head) {
  throw "Git identity changed after manifest creation"
}

$active = @(Get-CimInstance Win32_Process | Where-Object {
  $_.ProcessId -ne $PID -and
  ($_.Name -match '^(python|python3|mujoco|conda|pip|uv|aria2|curl|wget)' -or
   $_.CommandLine -match 'LIBERO|VLA|rollout|train|download|huggingface|mujoco') -and
  $_.CommandLine -notmatch 'invoke_windows_storage_cleanup.ps1|storage_cleanup_archive.py'
})
if ($active.Count) {
  throw "Relevant Windows worker/process detected: $($active | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress)"
}

$targets = @($Manifest.targets | Where-Object platform -eq "windows")
$validated = [Collections.Generic.List[object]]::new()
foreach ($target in $targets) {
  if ($target.classification -ne "VERIFIED_DISPOSABLE") { throw "Bad class: $($target.path)" }
  if (-not (Test-Path -LiteralPath $target.path)) { throw "Target disappeared: $($target.path)" }
  $item = Get-Item -LiteralPath $target.path -Force
  if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Top-level reparse target refused: $($target.path)" }
  $resolved = [IO.Path]::GetFullPath($item.FullName).TrimEnd('\')
  $expected = [IO.Path]::GetFullPath([string]$target.resolved_path).TrimEnd('\')
  $root = [IO.Path]::GetFullPath([string]$target.audited_root).TrimEnd('\')
  if (-not $resolved.Equals($expected, [StringComparison]::OrdinalIgnoreCase)) { throw "Resolved identity changed: $($target.path)" }
  if (-not ($resolved.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase))) { throw "Outside audited root: $resolved" }
  foreach ($p in $Protected) {
    $protectedPath = [IO.Path]::GetFullPath($p).TrimEnd('\')
    if ($resolved.Equals($protectedPath, [StringComparison]::OrdinalIgnoreCase) -or
        $resolved.StartsWith($protectedPath + '\', [StringComparison]::OrdinalIgnoreCase) -or
        $protectedPath.StartsWith($resolved + '\', [StringComparison]::OrdinalIgnoreCase)) {
      throw "Protected-path overlap: $resolved"
    }
  }
  if ($target.immutable_revision -and $resolved.StartsWith("C:\assets\repos\", [StringComparison]::OrdinalIgnoreCase)) {
    $repoStatus = @(git -C $resolved status --porcelain)
    $repoHead = git -C $resolved rev-parse HEAD
    if ($repoStatus.Count -or $repoHead -ne $target.immutable_revision) { throw "External repo identity changed: $resolved" }
  }
  $metrics = Get-TargetMetrics $resolved
  if ([int64]$metrics.apparent_size_bytes -ne [int64]$target.apparent_size_bytes -or
      [int64]$metrics.allocated_size_bytes -ne [int64]$target.allocated_size_bytes) {
    throw "Target size changed: $resolved"
  }
  if ($target.content_sha256) {
    $hash = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash
    if ($hash -ne $target.content_sha256) { throw "High-risk target hash changed: $resolved" }
  }
  $locks = @(Test-ExclusiveRead $resolved)
  if ($locks.Count) { throw "Open handles found under $resolved : $($locks | ConvertTo-Json -Compress)" }
  $validated.Add([pscustomobject]@{
    target_id = $target.target_id
    path = $resolved
    apparent_size_bytes = $metrics.apparent_size_bytes
    allocated_size_bytes = $metrics.allocated_size_bytes
    exclusive_handle_check = "PASS"
    status = "VALIDATED"
  })
}

$results = [Collections.Generic.List[object]]::new()
if ($Execute) {
  foreach ($entry in $validated) {
    Remove-Item -LiteralPath $entry.path -Recurse -Force -ErrorAction Stop
    if (Test-Path -LiteralPath $entry.path) { throw "Deletion verification failed: $($entry.path)" }
    $results.Add([pscustomobject]@{ target_id=$entry.target_id; path=$entry.path; status="DELETED"; allocated_size_bytes=$entry.allocated_size_bytes })
  }
} else {
  foreach ($entry in $validated) {
    $results.Add([pscustomobject]@{ target_id=$entry.target_id; path=$entry.path; status="DRY_RUN_VALIDATED"; allocated_size_bytes=$entry.allocated_size_bytes })
  }
}

$payload = [ordered]@{
  schema_version = "storage_cleanup.windows_execution.v1"
  timestamp = (Get-Date).ToString("o")
  execute = [bool]$Execute
  branch = $branch
  head = $head
  target_count = $results.Count
  allocated_size_bytes = ($results | Measure-Object allocated_size_bytes -Sum).Sum
  results = $results
}
$payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $LogPath -Encoding UTF8
[pscustomobject]@{
  timestamp = $payload.timestamp
  execute = $payload.execute
  target_count = $payload.target_count
  allocated_size_bytes = $payload.allocated_size_bytes
} | ConvertTo-Json
