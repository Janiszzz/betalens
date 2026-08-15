param(
    [Parameter(Mandatory = $true)]
    [int]$CoordinatorPid,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [int]$IntervalSeconds = 10
)

$ErrorActionPreference = 'Stop'
$output = [System.IO.Path]::GetFullPath($OutputPath)
$directory = [System.IO.Path]::GetDirectoryName($output)
[System.IO.Directory]::CreateDirectory($directory) | Out-Null

$header = 'timestamp,root_pid,coordinator_pid,python_process_count,total_cpu_percent,coordinator_cpu_percent,private_memory_mb,working_set_mb,available_memory_mb'
[System.IO.File]::WriteAllText($output, $header + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))

$previousCpu = @{}
$previousAt = Get-Date
while (Get-Process -Id $CoordinatorPid -ErrorAction SilentlyContinue) {
    $sampleAt = Get-Date
    $elapsed = [Math]::Max(($sampleAt - $previousAt).TotalSeconds, 0.001)
    $logicalProcessors = [Math]::Max([Environment]::ProcessorCount, 1)
    $processTable = @(Get-CimInstance Win32_Process)
    $descendantIds = @($CoordinatorPid)
    $frontier = @($CoordinatorPid)
    while ($frontier.Count -gt 0) {
        $children = @($processTable | Where-Object { $_.ParentProcessId -in $frontier } | Select-Object -ExpandProperty ProcessId)
        if ($children.Count -eq 0) { break }
        $descendantIds += $children
        $frontier = $children
    }
    $pythonIds = @(
        $processTable |
            Where-Object { $_.ProcessId -in $descendantIds -and $_.Name -in @('python.exe', 'pythonw.exe') } |
            Select-Object -ExpandProperty ProcessId
    )
    $actualCoordinatorId = @(
        $processTable |
            Where-Object { $_.ParentProcessId -eq $CoordinatorPid -and $_.Name -in @('python.exe', 'pythonw.exe') } |
            Select-Object -ExpandProperty ProcessId -First 1
    )
    if ($actualCoordinatorId.Count -eq 0) { $actualCoordinatorId = @($CoordinatorPid) }
    $actualCoordinatorId = [int]$actualCoordinatorId[0]
    $processes = @(Get-Process -Id $pythonIds -ErrorAction SilentlyContinue)
    $totalCpu = 0.0
    $coordinatorCpu = 0.0
    foreach ($process in $processes) {
        $current = [double]$process.CPU
        if ($previousCpu.ContainsKey($process.Id)) {
            $percent = 100.0 * ($current - [double]$previousCpu[$process.Id]) / $elapsed / $logicalProcessors
            $totalCpu += [Math]::Max($percent, 0.0)
            if ($process.Id -eq $actualCoordinatorId) { $coordinatorCpu = [Math]::Max($percent, 0.0) }
        }
        $previousCpu[$process.Id] = $current
    }
    $memory = Get-CimInstance Win32_OperatingSystem
    $line = '{0},{1},{2},{3},{4:F2},{5:F2},{6:F2},{7:F2},{8:F2}' -f @(
        $sampleAt.ToString('o'),
        $CoordinatorPid,
        $actualCoordinatorId,
        $processes.Count,
        $totalCpu,
        $coordinatorCpu,
        (($processes | Measure-Object PrivateMemorySize64 -Sum).Sum / 1MB),
        (($processes | Measure-Object WorkingSet64 -Sum).Sum / 1MB),
        ($memory.FreePhysicalMemory / 1KB)
    )
    [System.IO.File]::AppendAllText($output, $line + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
    $previousAt = $sampleAt
    Start-Sleep -Seconds $IntervalSeconds
}
