param(
    [Parameter(Mandatory = $true)]
    [string]$SourceFolder
)

if (-not (Test-Path -LiteralPath $SourceFolder -PathType Container)) {
    Write-Error "Folder does not exist: $SourceFolder"
    exit 1
}

$SourceFolder = (Resolve-Path -LiteralPath $SourceFolder).Path

function Get-TimestampFolderName {
    param(
        [string]$FileName
    )

    # Pattern 1: camera1_2026-03-18T15_09_00
    if ($FileName -match '(\d{4}-\d{2}-\d{2}T\d{2}_\d{2})_\d{2}') {
        return $matches[1]
    }

    # Pattern 2: open_loop_2_simutaneous_cube_mapping_20260318_150900
    if ($FileName -match '(\d{8})_(\d{6})') {
        $datePart = $matches[1]
        $timePart = $matches[2]

        $year   = $datePart.Substring(0, 4)
        $month  = $datePart.Substring(4, 2)
        $day    = $datePart.Substring(6, 2)

        $hour   = $timePart.Substring(0, 2)
        $minute = $timePart.Substring(2, 2)

        return "$year-$month-$day" + "T" + "$hour" + "_" + "$minute"
    }

    return $null
}

Get-ChildItem -LiteralPath $SourceFolder -File -Recurse | ForEach-Object {
    $file = $_
    $timestampFolder = Get-TimestampFolderName -FileName $file.Name

    if ($null -ne $timestampFolder) {
        $targetFolder = Join-Path $SourceFolder $timestampFolder

        if (-not (Test-Path -LiteralPath $targetFolder)) {
            New-Item -ItemType Directory -Path $targetFolder | Out-Null
        }

        $currentDir = $file.Directory.FullName

        if ($currentDir -ne $targetFolder) {
            $destinationPath = Join-Path $targetFolder $file.Name

            Move-Item -LiteralPath $file.FullName -Destination $destinationPath -Force

            Write-Host "Moved: $($file.Name) -> $timestampFolder/"
        }
    }
}

Write-Host "Done: Files grouped by timestamp (minute-level)."